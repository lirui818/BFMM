import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='BFMM')  
    parser.add_argument('--dataset', type=str, default='Cave', help='Lowercase word') 
    parser.add_argument("--trainset_sizeI", default=64, type=int, help='trainset_size')
    parser.add_argument("--sf", default=8, type=int, help='Scaling factor')
    parser.add_argument("--batch_size", default=16, type=int, help='Batch size')  
    parser.add_argument("--num_workers", default=4, type=int, help='Number of threads')  
    parser.add_argument('--epochs', type=int, default=300, help='End epoch for training')  
    parser.add_argument("--trainset_num", default=20000, type=int, help='The number of training samples of each epoch')  
    parser.add_argument("--test_model_name", default="model_0205", help='Test Model Epoch')

    parser.add_argument('--lr', type=float, default=3e-4)  # 3e-4  1e-4
    parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='ADAM beta2')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
    parser.add_argument("--seed", default=1, type=int, help='Random seed')
    parser.add_argument('--data_path', default='./data', type=str, help='Path of the data dir') 
    parser.add_argument('--save_path', default='./experiment', type=str, help='Path of the save dir')
    parser.add_argument('--result_path', default='./Result', type=str, help='Path of the result dir')
    args = parser.parse_args()
    return args