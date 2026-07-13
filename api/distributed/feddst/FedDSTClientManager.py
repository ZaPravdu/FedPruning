import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

try:
    from core.distributed.client.client_manager import ClientManager
    from core.distributed.communication.message import Message
except ImportError:
    from FedPruning.core.distributed.client.client_manager import ClientManager
    from FedPruning.core.distributed.communication.message import Message
from .message_define import MyMessage
from .utils import transform_list_to_tensor, post_complete_message_to_sweep_process


class FedDSTClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.mode = 0
        self._last_density = 1.0

    # mode 0: the client receieve both weight and mask from server;  config the new weight and mask, train; send the weight
    # mode 1: the client receieve weights;  config the new weight, train;  send the weight,
    # mode 2: the client receieve weights;  config the new weight; train weight and masks; send the weight and masks
    # TODO (special mode, only for \delta t == 1) mode 3: receieve both weight and mask; config the new weight and mask, train weight and masks; send the weight and masks

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        self.mode = msg_params.get(MyMessage.MSG_ARG_KEY_MODE_CODE)
        self.round_idx =  msg_params.get(MyMessage.MSG_ARG_KEY_ROUND_IDX)

        if self.args.is_mobile == 1:
            global_model_params = transform_list_to_tensor(global_model_params)

        self.trainer.update_model(global_model_params)
        self.trainer.update_dataset(int(client_index))

        # init weight_archive from server initial weights (dense)
        if getattr(self.args, "weight_archive", False):
            self.trainer.trainer.model.init_weight_archive()

        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)
        self.mode = msg_params.get(MyMessage.MSG_ARG_KEY_MODE_CODE)
        self.round_idx =  msg_params.get(MyMessage.MSG_ARG_KEY_ROUND_IDX)

        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(model_params)

        if self.mode in [0, 3]:
            mask_dict = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_MASKS)
            self.trainer.trainer.model.mask_dict = mask_dict
            self.trainer.trainer.model.apply_mask()

        self.trainer.update_model(model_params)
        self.trainer.update_dataset(int(client_index))

        # update weight_archive at mask==1 positions with fresh server weights
        if getattr(self.args, "weight_archive", False):
            self.trainer.trainer.model.update_weight_archive(
                self.trainer.trainer.model.mask_dict
            )

        self.__train()
        if self.round_idx == self.num_rounds:
            # write diagnostics JSON (client process)
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../experiments/feddst/results")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"density_debug_rank{self.get_sender_id()}.json")
            diag = getattr(self.trainer.trainer, "diagnostics", None)
            if diag:
                diag["rank"] = self.get_sender_id()
                diag["config"]["rounds_total"] = self.num_rounds
                with open(out_path, "w") as f:
                    json.dump(diag, f, indent=2, default=str)
                logging.info(f"[DIAGNOSTICS] client rank {self.get_sender_id()} JSON written to {out_path}")
            # post_complete_message_to_sweep_process(self.args)
            time.sleep(60)
            self.finish()

    def send_model_to_server(self, receive_id, weights, local_sample_num, density, masks=None):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_MASKS, masks)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        message.add_params(MyMessage.MSG_ARG_KEY_DENSITY, density)
        self.send_message(message)


    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        weights, masks, local_sample_num = self.trainer.train(mode = self.mode, round_idx=self.round_idx, )
        self._last_density = self.trainer.trainer.model.compute_gate_guided_density()
        if self.mode in [2, 3]:
            self.send_model_to_server(0, weights, local_sample_num, self._last_density, masks)
        else:
            self.send_model_to_server(0, weights, local_sample_num, self._last_density)
