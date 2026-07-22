import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
from api.data_preprocessing.svhn.data_loader import load_partition_data_svhn
from api.data_preprocessing.tinystories.data_loader import load_partition_data_tinystories
from api.data_preprocessing.tinyimagenet.data_loader import load_partition_data_tiny

from api.model.cv.resnet_gn import resnet18 as resnet18_gn
from api.model.cv.mobilenet import mobilenet
from api.model.cv.resnet import resnet18, resnet56
from api.model.nlp.gpt2 import GPT2Model, GPT2Config
from torchvision.models import mobilenet_v3_small as MobileNetV3
from torchvision.models import efficientnet_v2_s as EfficientNetV2
from torchvision.models import squeezenet1_1 as SqueezeNet
from torchvision.models import shufflenet_v2_x0_5 as ShuffleNet
from torchvision.models import swin_t as SwinT
from torchvision.models import vit_b_16 as ViT
from torchvision.models import mnasnet0_75 as MNASNet

from api.distributed.feddst.FedDSTAggregator import FedDSTAggregator
from api.distributed.feddst.FedDSTTrainer import FedDSTTrainer
from api.distributed.feddst.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from api.distributed.feddst.my_model_trainer_language_model import MyModelTrainer as MyModelTrainerLM
from api.pruning.model_pruning import SparseModel
from api.pruning.init_scheme import pruning


def add_args(parser):
    parser.add_argument("--model", type=str, default="resnet56", metavar="N",
                        help="neural network used in training, e.g. resnet18, resnet56")
    parser.add_argument("--dataset", type=str, default="cifar10", metavar="N", help="dataset used for training")
    parser.add_argument("--dataset_ratio", type=float, default=0.05, metavar="PA",
                        help="the ratio of subset for the total dataset (default: 0.05). Only appliable for [tinystories, ]")
    parser.add_argument("--partition_alpha", type=float, default=0.5, metavar="PA", help="partition alpha (default: 0.5)")
    parser.add_argument("--client_num_in_total", type=int, default=2, metavar="NN", help="number of workers")
    parser.add_argument("--client_num_per_round", type=int, default=2, metavar="NN", help="number of workers per round")
    parser.add_argument("--batch_size", type=int, default=64, metavar="N", help="input batch size for training")
    parser.add_argument("--nlp_hidden_size", type=int, default=256, metavar="N", help="the hidden size for nlp model")
    parser.add_argument("--num_eval", type=int, default=128, help="the number of data samples used for eval")
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument("--p", type=float, default=0.85, help="CDF top-p ratio for pruning (was --gate_p)")
    parser.add_argument("--adjustment_type", type=str, default=None, choices=["mag", "mag_cdf", "mag_grad_mag"],
                        help="pruning strategy in adjustment rounds (default: original prune+grow). options: mag | mag_cdf | mag_grad_mag")
    parser.add_argument("--cdf_pos", type=str, default="post-train", choices=["pre-train", "post-train"],
                        help="when to apply CDF pruning: pre-train (prune then train) or post-train (train then prune)")
    parser.add_argument("--epochs", type=int, default=1, metavar="EP", help="local epochs")
    parser.add_argument("--A_epochs", type=int, default=1, metavar="EP",
                        help="how many epochs will be trained before pruning and growing; default uses half of local epochs in adjustment rounds")
    parser.add_argument("--comm_round", type=int, default=2, help="communication rounds")
    parser.add_argument("--frequency_of_the_test", type=int, default=5, help="test frequency")
    parser.add_argument('--pruning_strategy', type=str, default="ERK_magnitude",
                        help='the distribution of layerwise density and the pruning method, options["uniform_magnitude", "ER_magnitude", "ERK_magnitude"]')
    parser.add_argument('--target_density', type=float, default=0.5, help='pruning target density')
    parser.add_argument('--delta_T', type=int, default=1, help='delta t for update')
    parser.add_argument('--T_end', type=int, default=100, help='end of time for update')
    parser.add_argument("--adjust_alpha", type=float, default=0.2, help='the ratio of num elements for adjustments')
    parser.add_argument("--adjustment_epochs", type=int, default=1,
                        help=" the number of local apoches used in model adjustment round, if it is set None, it is equal to the number of epoches for training round")
    parser.add_argument("--wd", help="weight decay parameter", type=float, default=0.001)
    parser.add_argument("--partition_method", type=str, default="hetero", help="how to partition dataset")
    parser.add_argument("--data_dir", type=str, default=None, help="data directory")
    parser.add_argument("--client_optimizer", type=str, default="sgd", help="optimizer")
    parser.add_argument("--growth_data_mode", type=str, default="batch",
                        help="the number of data samples used for parameter growth, option are [ 'random', 'single', 'batch', 'entire']")
    parser.add_argument("--is_mobile", type=int, default=0,
                        help="whether to transform tensor to list for mobile deployment (default: 0)")
    parser.add_argument('--init_density', type=float, default=None,
        help='ERK initialization density (default: equals target_density)')
    parser.add_argument(
        "--top_p_aggregate",
        action="store_true",
        default=False,
        help="server aggregates masks by vote frequency CDF top-p instead of OR + magnitude re-prune",
    )
    return parser.parse_args()


def load_data(args, dataset_name):
    if args.data_dir is None:
        if dataset_name == "tinyimagenet":
            args.data_dir = f"./../../data/Tiny-ImageNet"
        else:
            args.data_dir = f"./../../data/{dataset_name}"

    if dataset_name == "tinystories":
        dataset_tuple = load_partition_data_tinystories(
            args.partition_method, args.partition_alpha,
            args.client_num_in_total, args.batch_size, args.dataset_ratio
        )
    elif dataset_name == "tinyimagenet":
        dataset_tuple = load_partition_data_tiny(
            args.data_dir, args.partition_method, args.partition_alpha,
            args.client_num_in_total, args.batch_size
        )
    else:
        data_loaders = {
            "cifar10": load_partition_data_cifar10,
            "cifar100": load_partition_data_cifar100,
            "cinic10": load_partition_data_cinic10,
            "svhn": load_partition_data_svhn,
        }
        data_loader = data_loaders.get(dataset_name, load_partition_data_cifar10)
        dataset_tuple = data_loader(
            args.dataset, args.data_dir, args.partition_method,
            args.partition_alpha, args.client_num_in_total, args.batch_size
        )
    return dataset_tuple


def create_model(args, model_name, output_dim):
    logging.info(f"create_model. model_name = {model_name}, output_dim = {output_dim}")
    model = None

    if model_name == "resnet18_gn":
        model = resnet18_gn(num_classes=output_dim)
    elif model_name == "resnet18":
        model = resnet18(class_num=output_dim)
    elif model_name == "resnet56":
        model = resnet56(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    elif model_name == "mobilenetv3":
        model = MobileNetV3(num_classes=output_dim)
    elif model_name == "efficientnet":
        model = EfficientNetV2(num_classes=output_dim)
    elif model_name == "shufflenet":
        model = ShuffleNet(num_classes=output_dim)
    elif model_name == "squeezenet":
        model = SqueezeNet(num_classes=output_dim)
    elif model_name == "swint":
        model = SwinT(num_classes=output_dim)
    elif model_name == "vit":
        model = ViT(image_size=32, num_classes=output_dim)
    elif model_name == "mnasnet":
        model = MNASNet(num_classes=output_dim)
    elif model_name == "gpt2":
        GPT2Config["hidden_size"] = args.nlp_hidden_size
        model = GPT2Model(GPT2Config)
        logging.info(f"number of parameters: {model.get_num_params()/1e6:.2f}M")
    else:
        raise Exception(f"{model_name} is not found!")

    return model


def run_single_client(client_id, args, model, train_data_local_dict,
                     train_data_local_num_dict, device, round_idx):
    from copy import deepcopy

    local_model = deepcopy(model)
    local_model.to(device)

    if args.dataset in ["tinystories"]:
        model_trainer = MyModelTrainerLM(local_model, args.dataset)
    else:
        model_trainer = MyModelTrainerCLS(local_model)

    model_trainer.set_id(client_id)

    trainer = FedDSTTrainer(
        client_id,
        train_data_local_dict,
        train_data_local_num_dict,
        None,
        sum(train_data_local_num_dict.values()),
        device,
        args,
        model_trainer,
    )

    trainer.update_dataset(client_id)

    logging.info(f"Client {client_id} starting training with {train_data_local_num_dict[client_id]} samples")

    is_adjustment_round = round_idx % args.delta_T == 0 and round_idx > 0 and round_idx <= args.T_end

    if is_adjustment_round:
        mode = 2
    else:
        mode = 1

    weights, masks, local_sample_num = trainer.train(mode, round_idx)

    density = local_model.compute_gate_guided_density()

    return weights, masks, local_sample_num, density


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(f"Arguments: {args}")

    wandb.init(mode="disabled")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    dataset = load_data(args, args.dataset)
    (
        train_data_num, test_data_num, train_data_global,
        test_data_global, train_data_local_num_dict,
        train_data_local_dict, test_data_local_dict, class_num,
    ) = dataset

    inner_model = create_model(args, model_name=args.model, output_dim=class_num)
    model = SparseModel(inner_model, target_density=args.target_density, strategy=args.pruning_strategy)
    model.to(device)

    if args.dataset in ["tinystories"]:
        model_trainer = MyModelTrainerLM(model, args.dataset)
    else:
        model_trainer = MyModelTrainerCLS(model)
    model_trainer.set_id(-1)

    worker_num = args.client_num_per_round
    aggregator = FedDSTAggregator(
        train_data_global, test_data_global, train_data_num,
        train_data_local_dict, test_data_local_dict,
        train_data_local_num_dict, worker_num, device, args, model_trainer
    )

    logging.info(f"Starting FedDST simulation with {worker_num} clients per round for {args.comm_round} rounds")

    for round_idx in range(args.comm_round):
        logging.info(f"\n{'='*50}")
        logging.info(f"Communication Round {round_idx + 1}/{args.comm_round}")
        logging.info(f"{'='*50}\n")

        is_adjustment_round = round_idx % args.delta_T == 0 and round_idx > 0 and round_idx <= args.T_end
        logging.info(f"Adjustment round: {is_adjustment_round}")

        w_locals = []
        mask_locals = []
        local_sample_nums = []
        density_locals = []

        client_indices = list(range(args.client_num_in_total))[:args.client_num_per_round]

        for client_id in client_indices:
            try:
                weights, masks, sample_num, density = run_single_client(
                    client_id, args, model, train_data_local_dict,
                    train_data_local_num_dict, device, round_idx
                )
                w_locals.append(weights)
                mask_locals.append(masks)
                local_sample_nums.append(sample_num)
                density_locals.append(density)
                logging.info(f"Client {client_id} completed training")
            except Exception as e:
                logging.error(f"Client {client_id} failed: {str(e)}")
                raise

        for idx, (weights, sample_num, density) in enumerate(zip(w_locals, local_sample_nums, density_locals)):
            aggregator.add_local_trained_result(idx, weights, sample_num, density)

        if is_adjustment_round:
            for idx, mask in enumerate(mask_locals):
                aggregator.add_local_trained_mask(idx, mask)

        assert aggregator.check_whether_all_receive(), "Not all clients reported results"

        aggregated_weights = aggregator.aggregate()

        if is_adjustment_round:
            aggregated_mask = aggregator.aggregate_mask()

            layer_density_strategy, pruning_strategy = model.strategy.split("_")
            new_global_mask = pruning(model, model.current_layer_density_dict, pruning_strategy, mask_dict=aggregated_mask)
            model.mask_dict = new_global_mask
            model.to(device)
            model.apply_mask()

        aggregator.log_average_client_density(round_idx)
        aggregator.test_on_server_for_all_clients(round_idx)

        aggregator.log_sparsity_statistics(round_idx)

    logging.info("\nFedDST Training completed!")


if __name__ == "__main__":
    main()
