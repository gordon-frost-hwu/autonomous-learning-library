import numpy as np
from all.logging import ExperimentWriter
from .runner import SingleEnvRunner, ParallelEnvRunner, OptimisationRunner

class Experiment:
    def __init__(
            self,
            agents,
            envs,
            frames=np.inf,
            episodes=np.inf,
            render=False,
            quiet=False,
            write_loss=True,
    ):
        if not isinstance(agents, list):
            agents = [agents]

        if not isinstance(envs, list):
            envs = [envs]

        for env in envs:
            for agent in agents:
                if isinstance(agent, tuple):
                    agent_name = agent[0].__name__
                    runner = ParallelEnvRunner
                else:
                    agent_name = agent.__name__
                    runner = SingleEnvRunner

                runner(
                    agent,
                    env,
                    frames=frames,
                    episodes=episodes,
                    render=render,
                    quiet=quiet,
                    writer=self._make_writer(agent_name, env.name, write_loss),
                )

    def _make_writer(self, agent_name, env_name, write_loss):
        return ExperimentWriter(agent_name, env_name, write_loss)


class OptimisationExperiment:
    def __init__(
            self,
            agent,
            env,
            frames=np.inf,
            episodes=np.inf,
            render=False,
            quiet=False,
            log=True,
            write_loss=True,
            write_episode_return=False,
            writer=None,
    ):
        agent_name = agent.__name__
        self.runner = OptimisationRunner(
            agent,
            env,
            frames=frames,
            episodes=episodes,
            render=render,
            log=log,
            quiet=quiet,
            write_episode_return=write_episode_return,
            writer=writer
        )

    @property
    def log_dir(self):
        return self.runner.log_dir
