import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy
from rl_envs import VecEnv

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        # 定义world_model
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(
            config, self._wm, config.behavior_stop_grad
        )
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    # 调用：一个采样的步
    # Call: self._policy(), self._should_reset(), self._shouldtrain(), self._should_pretrain(), self._should_expl()
    # Input: obs, state, training
        # obs: only used in policy.
        # state: used for policy and dynamics, more information.
        # reset: whether the env is done.
    # Output: policy_output, state
    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if self._should_reset(step):
            state = None
        if state is not None and reset.any():
            mask = 1 - reset
            for key in state[0].keys():
                for i in range(state[0][key].shape[0]):
                    state[0][key][i] *= mask[i]
            for i in range(len(state[1])):
                state[1][i] *= mask[i]
        
        # training: 要记录各种log
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state


    # 策略：用给定的obs和state，输出policy和下一个state
    def _policy(self, obs, state, training):
        if state is None:
            batch_size = len(obs["image"])
            latent = self._wm.dynamics.initial(len(obs["image"]))
            action = torch.zeros((batch_size, self._config.num_actions)).to(
                self._config.device
            )
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(
            latent, action, embed, obs["is_first"], self._config.collect_dyn_sample
        )
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor_dist == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        action = self._exploration(action, training)
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    # 探索：用于给动作加噪声
    # expl_amount: action的采集数量
    # actor_dist: 是one_hot的话输出从OneHotDist中采样的结果，否则从Normal中采样
    # TODO: OneHotDist是不是采样出来的结果是one_hot的？
    def _exploration(self, action, training):
        amount = self._config.expl_amount if training else self._config.eval_noise
        if amount == 0:
            return action
        if "onehot" in self._config.actor_dist:
            probs = amount / self._config.num_actions + (1 - amount) * action
            return tools.OneHotDist(probs=probs).sample()
        else:
            return torch.clip(torchd.normal.Normal(action, amount).sample(), -1, 1)

    # 训练：从world model中获取metrics并存下来
    # TODO: 什么样的写法让self._task_behavior.train()的第二个参数是一个函数？
    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset

# 用于main的开环境步骤
def make_env(config, mode):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    elif suite == 'a1':
        # TODO: Unitree A1 locomotion environments
        raise NotImplementedError('Unitree A1 locomotion environments should not be used by Parallel.')
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env

# TODO: Change main() to DreamerRunner and training part to self.learn
# def main(config):

class DreamerRunner():
    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu',
                 history_length = 5,
                 ):
        self.config = train_cfg
        # === 0.0 固定每次实验结果（从种子和设备两个方面）===
        
        tools.set_seed_everywhere(self.config.seed)
        if self.config.deterministic_run:
            tools.enable_deterministic_run()
        
        # === 0.1 设定log记录目录以及训练|评估评率 ===
        # action_repeat是指将预测出的动作重复执行多少遍(想想MC)
        # 定义了logdir和logger等数据记录机制
        if not log_dir:
            self.logdir = pathlib.Path(self.config.logdir).expanduser()
        else:
            self.logdir = log_dir
            
        self.config.traindir = self.config.traindir or self.logdir / "train_eps"
        self.config.evaldir = self.config.evaldir or self.logdir / "eval_eps"
        self.config.steps //= self.config.action_repeat
        self.config.eval_every //= self.config.action_repeat
        self.config.log_every //= self.config.action_repeat
        self.config.time_limit //= self.config.action_repeat

        print("Logdir", self.logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.config.traindir.mkdir(parents=True, exist_ok=True)
        self.config.evaldir.mkdir(parents=True, exist_ok=True)
        step = count_steps(self.config.traindir)
        # step in logger is environmental step
        self.logger = tools.Logger(self.logdir, self.config.action_repeat * step)

        print("Create envs.")
        # === 1.0 加载数据集 ===
        # 如果有offline_traindir和offline_evaldir则从这两个目录加载数据集，否则从traindir和evaldir加载
        # TODO: 如果数据集下面没有文件，等待simulate去建立文件，这里的train_eps和eval_eps应该是空的
        if self.config.offline_traindir:
            directory = self.config.offline_traindir.format(**vars(self.config))
        else:
            directory = self.config.traindir
        self.train_eps = tools.load_episodes(directory, limit=self.config.dataset_size)
        if self.config.offline_evaldir:
            directory = self.config.offline_evaldir.format(**vars(self.config))
        else:
            directory = self.config.evaldir
        self.eval_eps = tools.load_episodes(directory, limit=1)
        
        # === 1.1 开环境 ===
        # 有Parallel和Damy(Dummy?)两种模式，实现在parallel.py中
        # make_env在dreamer.py中，mode in ['train','eval']
        # config.envs为环境数量
        # config.parallel为bool
        
        # # Here env is implied by VecEnv, Paralleled as a single environment.
        # make = lambda mode: make_env(self.config, mode)
        # train_envs = [make("train") for _ in range(self.config.envs)]
        # eval_envs = [make("eval") for _ in range(self.config.envs)]
        # if self.config.parallel:
        #     train_envs = [Parallel(env, "process") for env in train_envs]
        #     eval_envs = [Parallel(env, "process") for env in eval_envs]
        # else:
        #     train_envs = [Damy(env) for env in train_envs]
        #     eval_envs = [Damy(env) for env in eval_envs]
        
        self.train_envs = [Damy(env)]
        self.eval_envs = [Damy(env)]
        
        acts = self.train_envs[0].action_space
        self.config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

        # === 1.2 warmup,在数据集中预填充若干数据 ===
        # simulate的功能：用给定的agent在给定的环境中采集数据，然后存到给定的目录中，要传入agent和envs
        # random_agent是随机构造了一个agent，正式训练的时候agent是RL算法，这里是为了不干扰算法的训练
        state = None
        if not self.config.offline_traindir:
            prefill = max(0, self.config.prefill - count_steps(self.config.traindir))
            print(f"Prefill dataset ({prefill} steps).")
            
            # 根据环境是否为discrete构建random_actor
            if hasattr(acts, "discrete"):
                random_actor = tools.OneHotDist(
                    torch.zeros(self.config.num_actions).repeat(self.config.envs, 1)
                )
            else:
                random_actor = torchd.independent.Independent(
                    torchd.uniform.Uniform(
                        torch.Tensor(acts.low).repeat(self.config.envs, 1),
                        torch.Tensor(acts.high).repeat(self.config.envs, 1),
                    ),
                    1,
                )

            # 根据random_actor构建random_agent，三个参数仅为了匹配常规agent的接口，实际上输出action是随机的，logprob由对应action算得
            def random_agent(o, d, s):
                action = random_actor.sample()
                logprob = random_actor.log_prob(action)
                return {"action": action, "logprob": logprob}, None

            # 会在self.train_eps中补充simulate出的数据
            state = tools.simulate(
                random_agent,
                self.train_envs,
                self.train_eps,
                self.config.traindir,
                self.logger,
                limit=self.config.dataset_size,
                steps=prefill,
            )
            self.logger.step += prefill * self.config.action_repeat
            print(f"Logger: ({self.logger.step} steps).")
        
        print("Simulate agent.")


    def learn(self):
        # === 2.0 构建可以训练的智能体dreamer ===
        train_dataset = make_dataset(self.train_eps, self.config)
        eval_dataset = make_dataset(self.eval_eps, self.config)
        # In Dreamer: __init__(self, obs_space, act_space, config, logger, dataset):
        agent = Dreamer(
            self.train_envs[0].observation_space,
            self.train_envs[0].action_space,
            self.config,
            self.logger,
            train_dataset,
        ).to(self.config.device)
        agent.requires_grad_(requires_grad=False)
        
        # 断点续传
        if (self.logdir / "latest.pt").exists():
            checkpoint = torch.load(self.logdir / "latest.pt")
            agent.load_state_dict(checkpoint["agent_state_dict"])
            tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
            agent._should_pretrain._once = False

        # make sure eval will be executed once after config.steps
        # 一共训练steps步，每eval_every步进行一次评估，一定最后进行一次评估（上一行注释的含义）
        while agent._step < self.config.steps + self.config.eval_every:
            self.logger.write()
            
            # === 2.1 先进行评估 ===
            # eval_episode_num个episode (常用值: 10)     
            if self.config.eval_episode_num > 0:
                print("Start evaluation.")
                eval_policy = functools.partial(agent, training=False)
                tools.simulate(
                    eval_policy,
                    self.eval_envs,
                    self.eval_eps,
                    self.config.evaldir,
                    self.logger,
                    is_eval=True,
                    episodes=self.config.eval_episode_num,
                )
                if self.config.video_pred_log:
                    video_pred = agent._wm.video_pred(next(eval_dataset))
                    self.logger.video("eval_openl", to_np(video_pred))
            
            # === 2.2 再进行训练 ===
            # 每次训练evel_every步(常用值:1e4)
            # TODO: 传梯度在什么地方？
            print("Start training.")
            state = tools.simulate(
                agent,
                self.train_envs,
                self.train_eps,
                self.config.traindir,
                self.logger,
                limit=self.config.dataset_size,
                steps=self.config.eval_every,
                state=state,
            )
            
            # === 2.3 保存模型 ===
            items_to_save = {
                "agent_state_dict": agent.state_dict(),
                "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            }
            torch.save(items_to_save, self.logdir / "latest.pt")
        
        self.close()
    
    def close(self):
        # === 3.0 关闭环境 ===
        for env in self.train_envs + self.eval_envs:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    # 把update按层级结构赋值给base，让base变成字典套字典
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    # 解析命令行传入的参数和configs.yaml中的参数，将其合并到defaults中
    # 这段代码以后要好好学一学，每次都可以用
    # TODO: args.configs是什么？是命令行传入的参数吗？
    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
        
    # 调 用 主 函 数
    # main(parser.parse_args(remaining))
