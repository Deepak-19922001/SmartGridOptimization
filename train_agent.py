import os
from src.environment import SmartGridEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


def train_agent():

    # --- 1. Create Directories ---
    # Directory to save the trained models
    models_dir = "saved_models"
    # Directory for logging during training
    log_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # --- 2. Instantiate the Environment ---
    # The environment is wrapped in DummyVecEnv to make it compatible with stable-baselines3
    env = DummyVecEnv([lambda: SmartGridEnv(data_path='/Users/deepakkhandelwal/PycharmProjects/SmartGridSimulator/data/simulated_grid_data.csv')])

    # --- 3. Set up the PPO Agent ---
    # We are using the 'MlpPolicy' which is a standard multi-layer perceptron policy.
    # `verbose=1` will print training progress.
    # `tensorboard_log` will save logs for visualization with TensorBoard.
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        tensorboard_log=log_dir
    )

    # --- 4. Set up Callbacks for Better Training ---
    # Stop training when the model achieves a certain average reward
    # Note: The 'reward_threshold' is an estimate. You might need to adjust it based on training performance.
    # A positive reward means the agent is making more money than it's spending.
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)

    # The EvalCallback saves the best model found during training
    eval_callback = EvalCallback(
        env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=1000,  # Run evaluation every 1000 steps
        callback_on_new_best=callback_on_best,
        verbose=1
    )

    # --- 5. Train the Agent ---
    # TOTAL_TIMESTEPS defines how long the training will run.
    # 50,000 is a good starting point for a reasonable training session.
    # For better performance, you would typically use a much higher number (e.g., 1,000,000).
    TOTAL_TIMESTEPS = 50000

    print("--- Starting Agent Training ---")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True
    )
    print("--- Agent Training Complete ---")

    # --- 6. Save the Final Model ---
    # The EvalCallback already saves the best model, but we'll save the final one too.
    final_model_path = os.path.join(models_dir, "ppo_smart_grid_final")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")


if __name__ == '__main__':
    train_agent()

