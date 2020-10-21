from __main__ import *

R_buffer, R_avg, eps, avg_wins, i, ep_reward, R_avg_progress = main.train_loop_ddqn(ddqn, env, replay_buffer, num_episodes, enable_visualization=enable_visualization, batch_size=batch_size, gamma=gamma, eps=eps, eps_end=eps_end, eps_decay=eps_decay)




while True
    username = input("Continue?: ")


result_dict["epsilon"].append(eps)
result_dict["avg_wins"].append(avg_wins)
result_dict["episodes"].append(i)
result_dict["ep_reward"].append(ep_reward)
result_dict["running_average"].append(R_avg)
result_dict["boxes_left"].append(R_avg_progress)


plot_reward(result_dict)

write_to_json(result_dict, filepath)
#data = load_from_json(filepath)
