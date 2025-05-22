import numpy as np
import os
import scipy.io as sio

root_dir = "dataset/amigo/raw_data"

def subject2video():
    # Only use 16 short videos
    video_data = [[np.array([]) for _ in range(14)] for _ in range(16)]

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.mat'):

                mat_file_path = os.path.join(dirpath, filename)
            
                print(f"找到 .mat 文件: {mat_file_path}")
                
                all_data = sio.loadmat(mat_file_path)
                signal_data = all_data['joined_data'][0]
                
                for video_index in range(16):
                    for channel_index in range(14):                    
                        channel_data = signal_data[video_index][:, channel_index]
                        if video_data[video_index][channel_index].size == 0:
                            video_data[video_index][channel_index] = channel_data
                        else:
                            video_data[video_index][channel_index] = np.concatenate(
                                (video_data[video_index][channel_index], channel_data)
                            )
    
    save_dir = "dataset/amigo/videos"
    os.makedirs(save_dir, exist_ok=True)
    for video_index in range(16):
        # Save each video data to a .mat file
        video_file_path = os.path.join("dataset/amigo/videos", f"video_{video_index + 1}.mat")
        sio.savemat(video_file_path, {'video_data': video_data[video_index]})
        print(f"已保存视频 {video_index + 1} 到 {video_file_path}")

if __name__ == "__main__":
    subject2video()