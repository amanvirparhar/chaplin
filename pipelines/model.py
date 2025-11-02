#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import json
import torch
import argparse
import numpy as np

from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import add_results_to_json
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E


class AVSR(torch.nn.Module):
    def __init__(self, modality, model_path, model_conf, rnnlm=None, rnnlm_conf=None,
        penalty=0., ctc_weight=0.1, lm_weight=0., beam_size=40, device="cuda:0"):
        super(AVSR, self).__init__()
        self.device = device

        if modality == "audiovisual":
            from espnet.nets.pytorch_backend.e2e_asr_transformer_av import E2E
        else:
            from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E

        with open(model_conf, "rb") as f:
            confs = json.load(f)
        args = confs if isinstance(confs, dict) else confs[2]
        self.train_args = argparse.Namespace(**args)

        labels_type = getattr(self.train_args, "labels_type", "char")
        if labels_type == "char":
            self.token_list = self.train_args.char_list
        elif labels_type == "unigram5000":
            file_path = os.path.join(os.path.dirname(__file__), "tokens", "unigram5000_units.txt")
            with open(file_path, encoding='utf-8') as f:
                self.token_list = ['<blank>'] + [word.split()[0] for word in f.read().splitlines()] + ['<eos>']
        self.odim = len(self.token_list)

        self.model = E2E(self.odim, self.train_args)
        self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        self.model.to(device=self.device).eval()

        self.beam_search = get_beam_search_decoder(self.model, self.token_list, rnnlm, rnnlm_conf, penalty, ctc_weight, lm_weight, beam_size)
        self.beam_search.to(device=self.device).eval()
        
    def infer(self, data):
        with torch.no_grad():
            if isinstance(data, tuple):
                enc_feats = self.model.encode(data[0].to(self.device), data[1].to(self.device))
            else:
                enc_feats = self.model.encode(data.to(self.device))
            nbest_hyps = self.beam_search(enc_feats)
            nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
            transcription = add_results_to_json(nbest_hyps, self.token_list)
            transcription = transcription.replace("▁", " ").strip()
        return transcription.replace("<eos>", "")


def get_beam_search_decoder(model, token_list, rnnlm=None, rnnlm_conf=None, penalty=0, ctc_weight=0.1, lm_weight=0., beam_size=40):
    sos = model.odim - 1
    eos = model.odim - 1
    scorers = model.scorers()

    if not rnnlm:
        lm = None
    else:
        lm_args = get_model_conf(rnnlm, rnnlm_conf)
        lm_model_module = getattr(lm_args, "model_module", "default")
        lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
        lm = lm_class(len(token_list), lm_args)
        torch_load(rnnlm, lm)
        lm.eval()

    scorers["lm"] = lm
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = dict(
        decoder=1.0 - ctc_weight,
        ctc=ctc_weight,
        lm=lm_weight,
        length_bonus=penalty,
    )

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )
