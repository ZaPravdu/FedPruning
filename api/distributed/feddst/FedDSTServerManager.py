import json
import logging
import os, signal
import sys

from .message_define import MyMessage
from .utils import transform_tensor_to_list, post_complete_message_to_sweep_process
from api.pruning.init_scheme import cubic_density_schedule, generate_layer_density_dict, pruning, cdf_prune_by_metric
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
try:
    from core.distributed.communication.message import Message
    from core.distributed.server.server_manager import ServerManager
except ImportError:
    from FedPruning.core.distributed.communication.message import Message
    from FedPruning.core.distributed.server.server_manager import ServerManager

class FedDSTServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="MPI", is_preprocessed=False, preprocessed_client_lists=None):
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists
        self.mode = 0 

        # mode 0, the server send both weight and mask to clients, received the weight, perform weight aggregation, if t % \delta t == 0 and t <= t_end, go to mode 2, else, go to mode 1
        # mode 1, the server send weights, received the weights, perform weights aggregation, if t % \delta t == 0 and t <= t_end, go to mode 2, else, go to mode 1
        # mode 2, the server send weights, received the weights and masks, perform weights and masks aggregation, pruning and growing to produce new mask,  go to mode 0
        # TODO (special mode, only for \delta t == 1) mode 3, the server send both weight and mask to clients, received the weights and masks, perform weights and masks aggregation, pruning and growing to produce new mask,  if t < t_end, go to mode 3 ,else , go to mode 0.

    def run(self):
        super().run()

    def send_init_msg(self):
        # sampling clients

        logging.info(f"current step is {self.round_idx} and the current mode is {self.mode}")
        client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                         self.args.client_num_per_round)
        global_model_params = self.aggregator.get_global_model_params()
        if self.args.is_mobile == 1:
            global_model_params = transform_tensor_to_list(global_model_params)
        for process_id in range(1, self.size):
            self.send_message_init_config(process_id, global_model_params, client_indexes[process_id - 1], self.mode, self.round_idx)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                self.handle_message_receive_model_from_client)

    def mode_convert(self,):
        if self.mode == 0:
            if self.round_idx % self.args.delta_T == 0 and self.round_idx <= self.args.T_end :
                self.mode = 2
            else:
                self.mode = 1
        elif self.mode == 1:
            if self.round_idx % self.args.delta_T == 0 and self.round_idx <= self.args.T_end :
                self.mode = 2
            else:
                self.mode = 1
        elif self.mode == 2:
            self.mode = 0
        elif self.mode == 3:
            if self.round_idx < self.args.T_end :
                self.mode = 3
            else:
                self.mode = 0

        return self.mode
    
    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        density = msg_params.get(MyMessage.MSG_ARG_KEY_DENSITY)
        if self.mode in [2, 3]:
            masks = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_MASKS)
            self.aggregator.add_local_trained_mask(sender_id - 1, masks)

        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number, density)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            logging.info(f"[MODE_DEBUG] round={self.round_idx} mode={self.mode} "
                         f"delta_T={self.args.delta_T} T_end={self.args.T_end} "
                         f"comm_round={self.args.comm_round}")
            self.aggregator.log_average_client_density(self.round_idx)
            global_model_params = self.aggregator.aggregate()
            logging.info(f"current mode for server is {self.mode}, the round is {self.round_idx}")
            if self.mode in [2, 3]:
                logging.info(f"[SCHED_CHECK] density_scheduler={self.args.density_scheduler} type={type(self.args.density_scheduler)} "
                             f"target_density={self.args.target_density} strategy={self.args.pruning_strategy}")
                model = self.aggregator.trainer.model

                # ── top_p_aggregate: frequency-based CDF top p (ignores density scheduler) ──
                if getattr(self.args, "top_p_aggregate", False):
                    if self.args.density_scheduler is not None:
                        logging.warning("[TOP_P_AGGREGATE] density_scheduler set but ignored — "
                                        "top_p_aggregate does not target a specific density")
                    candidate = self.aggregator.aggregate_mask()
                    freq_dict = self.aggregator.aggregate_mask_frequency()
                    for k in candidate.keys():
                        min_keep = None
                        if k in model.layer_density_dict:
                            min_keep = int(candidate[k].numel() * model.layer_density_dict[k])
                        metric = freq_dict.get(k)
                        if metric is not None:
                            candidate[k] = cdf_prune_by_metric(
                                metric, model.model.get_parameter(k),
                                candidate[k], p=self.args.p, min_keep=min_keep,
                            )
                    model.mask_dict = candidate
                    logging.info(f"[TOP_P_AGGREGATE] round={self.round_idx} p={self.args.p}")

                else:
                    # ── original aggregation ──
                    # density scheduler: update current_density / current_layer_density_dict
                    if self.args.density_scheduler is not None:
                        start_density = self.args.init_density if self.args.init_density is not None else self.args.target_density
                        new_current_density = cubic_density_schedule(
                            self.round_idx, self.args.density_scheduler[1],
                            start_density, self.args.density_scheduler[0],
                        )
                        layer_density_strategy, _ = model.strategy.split("_")
                        model.current_density = new_current_density
                        model.current_layer_density_dict = generate_layer_density_dict(
                            model.layer_shape_dict, model.num_overall_elements,
                            model.sparse_layer_set, new_current_density, layer_density_strategy,
                        )
                        logging.info(f"[DENSITY_SCHED] round={self.round_idx} target={new_current_density:.4f} "
                                     f"dense_ratio={model.num_overall_elements:.0f} "
                                     f"layer_densities={ {k: f'{v:.3f}' for k, v in model.current_layer_density_dict.items()} }")

                    global_mask = self.aggregator.aggregate_mask()
                    # ── diagnostic: OR mask density ──
                    if global_mask:
                        or_ones = sum((v != 0).sum().item() for v in global_mask.values())
                        or_total = sum(v.numel() for v in global_mask.values())
                        self.aggregator.diagnostics["or_masks"].append({
                            "round": self.round_idx,
                            "or_density": or_ones / max(or_total, 1),
                            "n_clients": len([k for k in self.aggregator.mask_dict if k is not None]),
                        })
                    # ─────────────────────────────────
                    # CDF lock: skip density reset, use aggregated mask as-is
                    # (layer_density_dict floor is always present since v2 refactor)
                    if getattr(self.args, "adjustment_type", None) is not None:
                        model.mask_dict = global_mask
                        logging.info("[CDF_LOCK] skipping density reset, using aggregated mask")
                    else:
                        # prune to reach density (always resets to current density)
                        layer_density_strategy, pruning_strategy = model.strategy.split("_")
                        new_global_mask = pruning(model, model.current_layer_density_dict, pruning_strategy, mask_dict=global_mask)
                        model.mask_dict = new_global_mask

                # model.mask_dict = global_mask
                model.to(self.aggregator.device)
                if not getattr(self.args, "mask_for_comm", False):
                    model.apply_mask()
                else:
                    logging.info("[MASK_FOR_COMM] skipping server apply_mask — weights kept dense for inference")

            # ── diagnostic: server density before logging ──
            server_d = self.aggregator.trainer.model.compute_gate_guided_density()
            self.aggregator.diagnostics["server_densities"].append({
                "round": self.round_idx,
                "mode": self.mode,
                "density": server_d,
            })
            # ────────────────────────────────────────────────
            self.aggregator.log_sparsity_statistics(self.round_idx)
            self.aggregator.log_communication_cost(self.round_idx, self.args.client_num_per_round)

            # logging.info("mask_dict after pruning and growing = " +str(mask_dict))
            kwargs = {"apply_mask": not getattr(self.args, "mask_for_comm", False)}
            self.aggregator.test_on_server_for_all_clients(self.round_idx, **kwargs)
            
            # start the next round
            self.round_idx += 1

            # convert the mode 
            self.mode = self.mode_convert()

            if self.round_idx == self.round_num + 1:
                # write diagnostics JSON (server process)
                out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../experiments/feddst/results")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"density_debug_server.json")
                self.aggregator.diagnostics["rank"] = 0
                self.aggregator.diagnostics["config"]["rounds_total"] = self.round_num
                self.aggregator.diagnostics["config"]["clients_per_round"] = self.args.client_num_per_round
                with open(out_path, "w") as f:
                    json.dump(self.aggregator.diagnostics, f, indent=2, default=str)
                logging.info(f"[DIAGNOSTICS] server JSON written to {out_path}")
                # post_complete_message_to_sweep_process(self.args)
                self.finish()
                print('here')
                return
            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    # sampling has already been done in data preprocessor
                    client_indexes = [self.round_idx] * self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.round_idx]
            else:
                # sampling clients
                client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                    self.args.client_num_per_round)

            print('indexes of clients: ' + str(client_indexes))
            print("size = %d" % self.size)
            logging.info(f"current step is {self.round_idx} and the current mode is {self.mode}")
            if self.args.is_mobile == 1:
                global_model_params = transform_tensor_to_list(global_model_params)
            
            if self.mode in [0, 3]:
                mask_dict = self.aggregator.trainer.model.mask_dict
                for k in mask_dict:
                    mask_dict[k] = mask_dict[k].cpu()
                for receiver_id in range(1, self.size):
                    self.send_message_sync_model_to_client(receiver_id, global_model_params,
                        client_indexes[receiver_id - 1], self.mode, self.round_idx, mask_dict)
            else:
                for receiver_id in range(1, self.size):
                    self.send_message_sync_model_to_client(receiver_id, global_model_params,
                        client_indexes[receiver_id - 1], self.mode, self.round_idx)


    def send_message_init_config(self, receive_id, global_model_params, client_index, mode_code, round_idx):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND_IDX, round_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_MODE_CODE, mode_code)
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index, mode_code, round_idx, mask_dict=None):
        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_ROUND_IDX, round_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_MODE_CODE, mode_code)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_MASKS, mask_dict)
        self.send_message(message)
