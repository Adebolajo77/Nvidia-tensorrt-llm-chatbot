import argparse
import ast
import csv
from pathlib import Path
import gc

import numpy as np
import torch
from utils import (DEFAULT_HF_MODEL_DIRS, DEFAULT_PROMPT_TEMPLATES,
                   load_tokenizer, read_model_name, throttle_generator)

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


class TRTLLM_API():
    def __init__(self,
                 engine_dir = None,
                 token_dir = None,
                 temperature = 1.0,
                 max_input_len = None,
                 max_output_len = None,
                 max_attention_window_size = None,
                 use_py_session = False,
                 streaming_interval = 5,
                 max_batch_size = None
                 ) -> None:


        ## input parameters/Arguments
        self.engine_dir = engine_dir
        self.tokenizer_dir = token_dir
        self.temperature = temperature
        self.use_py_session = use_py_session
        self.streaming_interval = streaming_interval
        self.max_batch_size = max_batch_size
        self.max_output_len = max_output_len
        self.max_input_len = max_input_len
        self.max_attention_window_size = max_attention_window_size
        

        # Configuration members
        self.initialize_config_members()


        self.log_level = 'error'
        # Initialize logging and model information
        self.runtime_rank = tensorrt_llm.mpi_rank()
        logger.set_level(self.log_level)
        # get model, names and version from the compilation file.
        self.model_name, self.model_version = read_model_name(self.engine_dir)


        self.tokenizer_type = None
        self.vocab_file = None
        # Load tokenizer
        self.tokenizer, self.pad_id, self.end_id = self.load_tokenizer()
        

        self.use_prompt_template = True
        # Set prompt template if applicable
        self.set_prompt_template()


        self.use_py_session = True
        self.debug_mode = False
        # Adjust session usage based on availability and debug mode
        self.adjust_session_usage()



        self.lora_dir = None
        self.lora_ckpt_source = "hf"
        self.gpu_weights_percent = 1.0
        self.medusa_choices = None
        self.num_beams = 1
        self.sink_token_length = None
        self.runner_kwargs = None
        # Initialize the model runner
        self.runner  = self.initialize_model_runner()

    def initialize_config_members(self):
        
        self.stop_words_list = None
        self.bad_words_list = None

        self.prompt_template = None
        self.num_prepend_vtokens = []
        self.add_special_tokens = True

        self.length_penalty = 1.0

        self.repetition_penalty = 1.0
        self.presence_penalty = 0.0
        self.frequency_penalty = 0.0
        self.output_cum_log_probs_npy = None
        self.output_log_probs_npy = None
        self.lora_task_uids = None
        self.prompt_table_path = None
        self.prompt_tasks = None

        self.output_csv = None
        self.output_npy = None
        self.output_logits_npy = None

    def load_tokenizer(self):
        if self.tokenizer_dir is None:
            logger.warning("tokenizer_dir is not specified. Try to infer from model_name, but this may be incorrect.")
            raise ValueError("Invalid token directory")
        return load_tokenizer(
            tokenizer_dir=self.tokenizer_dir,
            vocab_file=self.vocab_file,
            model_name=self.model_name,
            model_version=self.model_version,
            tokenizer_type=self.tokenizer_type,
        )


    def adjust_session_usage(self):
        if not PYTHON_BINDINGS and not self.use_py_session:
            logger.warning("Python bindings of C++ session is unavailable, fallback to Python session.")
            self.use_py_session = True
        if self.debug_mode and not self.use_py_session:
            logger.warning("Debug mode is not supported in C++ session for now, fallback to Python session.")
            self.use_py_session = True

    def set_prompt_template(self):
        if self.use_prompt_template and self.model_name in DEFAULT_PROMPT_TEMPLATES:
            self.prompt_template = DEFAULT_PROMPT_TEMPLATES[self.model_name]

    def initialize_model_runner(self):
        # create an instance of the model runner 
        runner_cls = ModelRunner if self.use_py_session else ModelRunnerCpp
        self.runner_kwargs = dict(engine_dir=self.engine_dir,
                                  lora_dir=self.lora_dir,
                                  rank=self.runtime_rank,
                                  debug_mode=self.debug_mode,
                                  lora_ckpt_source=self.lora_ckpt_source,
                                  gpu_weights_percent=self.gpu_weights_percent)
        
        if self.medusa_choices is not None:
            self.medusa_choices = ast.literal_eval(self.medusa_choices)
            assert self.temperature == 1.0, "Medusa should use temperature == 1.0"
            assert self.num_beams == 1, "Medusa should use num_beams == 1"
            self.runner_kwargs.update(medusa_choices=self.medusa_choices)
        
        if not self.use_py_session:
            self.runner_kwargs.update(
                max_batch_size=self.max_batch_size,
                max_input_len=self.max_input_len,
                max_output_len=self.max_output_len,
                max_beam_width=self.num_beams,
                max_attention_window_size=self.max_attention_window_size,
                sink_token_length=self.sink_token_length,
            )
        
        # return the loaded the model
        return runner_cls.from_dir(**self.runner_kwargs)

    def parse_input(self,
                tokenizer,
                input_text=None,
                prompt_template=None,
                input_file=None,
                add_special_tokens=True,
                max_input_length=923,
                pad_id=None,
                num_prepend_vtokens=[],
                model_name=None,
                model_version=None):
        if pad_id is None:
            pad_id = tokenizer.pad_token_id

        batch_input_ids = []
        if input_file is None:
            for curr_text in input_text:
                if prompt_template is not None:
                    curr_text = prompt_template.format(input_text=curr_text)
                input_ids = tokenizer.encode(curr_text,
                                            add_special_tokens=add_special_tokens,
                                            truncation=True,
                                            max_length=max_input_length)
                batch_input_ids.append(input_ids)
        else:
            if input_file.endswith('.csv'):
                with open(input_file, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    for line in csv_reader:
                        input_ids = np.array(line, dtype='int32')
                        batch_input_ids.append(input_ids[-max_input_length:])
            elif input_file.endswith('.npy'):
                inputs = np.load(input_file)
                for row in inputs:
                    input_ids = row[row != pad_id]
                    batch_input_ids.append(input_ids[-max_input_length:])
            elif input_file.endswith('.txt'):
                with open(input_file, 'r', encoding='utf-8',
                        errors='replace') as txt_file:
                    input_text = txt_file.readlines()
                    batch_input_ids = tokenizer(
                        input_text,
                        add_special_tokens=add_special_tokens,
                        truncation=True,
                        max_length=max_input_length)["input_ids"]
            else:
                print('Input file format not supported.')
                raise SystemExit

        if num_prepend_vtokens:
            assert len(num_prepend_vtokens) == len(batch_input_ids)
            base_vocab_size = tokenizer.vocab_size - len(
                tokenizer.special_tokens_map.get('additional_special_tokens', []))
            for i, length in enumerate(num_prepend_vtokens):
                batch_input_ids[i] = list(
                    range(base_vocab_size,
                        base_vocab_size + length)) + batch_input_ids[i]

        if model_name == 'ChatGLMForCausalLM' and model_version == 'glm':
            for ids in batch_input_ids:
                ids.append(tokenizer.sop_token_id)

        batch_input_ids = [
            torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
        ]
        return batch_input_ids

    def print_output(self,
                     tokenizer,
                    output_ids,
                    input_lengths,
                    sequence_lengths,
                    output_csv=None,
                    output_npy=None,
                    context_logits=None,
                    generation_logits=None,
                    cum_log_probs=None,
                    log_probs=None,
                    output_logits_npy=None,
                    output_cum_log_probs_npy=None,
                    output_log_probs_npy=None):
        batch_size, num_beams, _ = output_ids.size()
        if output_csv is None and output_npy is None:
            for batch_idx in range(batch_size):
                inputs = output_ids[batch_idx][0][:input_lengths[batch_idx]].tolist(
                )
                input_text = tokenizer.decode(inputs)
                #print(f'Input [Text {batch_idx}]: \"{input_text}\"')
                for beam in range(num_beams):
                    output_begin = input_lengths[batch_idx]
                    output_end = sequence_lengths[batch_idx][beam]
                    outputs = output_ids[batch_idx][beam][
                        output_begin:output_end].tolist()
                    output_text = tokenizer.decode(outputs)
                    #print("------------------------")
                    #print(
                    #    f'Output [Text {batch_idx} Beam {beam}]: \"{output_text}\"')
                    
                    
                    


        output_ids = output_ids.reshape((-1, output_ids.size(2)))
        

        if output_csv is not None:
            output_file = Path(output_csv)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            outputs = output_ids.tolist()
            with open(output_file, 'w') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerows(outputs)

        if output_npy is not None:
            output_file = Path(output_npy)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            outputs = np.array(output_ids.cpu().contiguous(), dtype='int32')
            np.save(output_file, outputs)

        # Save context logits
        if context_logits is not None and output_logits_npy is not None:
            context_logits = torch.cat(context_logits, axis=0)
            vocab_size_padded = context_logits.shape[-1]
            context_logits = context_logits.reshape([1, -1, vocab_size_padded])

            output_context_logits_npy = output_logits_npy.split(
                '.npy')[0] + "_context"
            output_context_logits_file = Path(output_context_logits_npy)
            context_outputs = np.array(
                context_logits.squeeze(0).cpu().contiguous(),
                dtype='float32')  # [promptLengthSum, vocabSize]
            np.save(output_context_logits_file, context_outputs)

        # Save generation logits
        if generation_logits is not None and output_logits_npy is not None and num_beams == 1:
            output_generation_logits_npy = output_logits_npy.split(
                '.npy')[0] + "_generation"
            output_generation_logits_file = Path(output_generation_logits_npy)
            generation_outputs = np.array(generation_logits.cpu().contiguous(),
                                        dtype='float32')
            np.save(output_generation_logits_file, generation_outputs)

        # Save cum log probs
        if cum_log_probs is not None and output_cum_log_probs_npy is not None:
            cum_log_probs_file = Path(output_cum_log_probs_npy)
            cum_log_probs_outputs = np.array(cum_log_probs.cpu().contiguous(),
                                            dtype='float32')
            np.save(cum_log_probs_file, cum_log_probs_outputs)

        # Save cum log probs
        if log_probs is not None and output_log_probs_npy is not None:
            log_probs_file = Path(output_log_probs_npy)
            log_probs_outputs = np.array(log_probs.cpu().contiguous(),
                                        dtype='float32')
            np.save(log_probs_file, log_probs_outputs)

        return output_text

    def generate(self,
                 input_text,
                 streaming=False,
                 streaming_interval = 2,
                 temperature = 1.0,
                 max_output_len = None,
                 top_k = 1,
                 top_p = 0.0,
                 early_stopping = 1,
                 input_file = None
                 ):


        if self.runner is not None:
        
            self.input_text = [input_text]

            
            batch_input_ids = self.parse_input(
                                    tokenizer=self.tokenizer,
                                    input_text=self.input_text,
                                    prompt_template=self.prompt_template,
                                    input_file=input_file,
                                    add_special_tokens=self.add_special_tokens,
                                    max_input_length=self.max_input_len,
                                    pad_id=self.pad_id,
                                    num_prepend_vtokens=self.num_prepend_vtokens,
                                    model_name=self.model_name,
                                    model_version=self.model_version)
            
            input_lengths = [x.size(0) for x in batch_input_ids]

            with torch.no_grad():
                outputs = self.runner.generate(
                    batch_input_ids,
                    max_new_tokens=max_output_len,
                    temperature = temperature,
                    streaming=streaming,
                    max_attention_window_size=self.max_attention_window_size,
                    top_k=top_k,
                    top_p=top_p,
                    early_stopping=early_stopping,
                    num_beams=self.num_beams,
                    output_sequence_lengths=True,
                    return_dict=True,

                    sink_token_length=self.sink_token_length,
                    end_id=self.end_id,
                    pad_id=self.pad_id,
                    length_penalty=self.length_penalty,
                    repetition_penalty=self.repetition_penalty,
                    presence_penalty=self.presence_penalty,
                    frequency_penalty=self.frequency_penalty,
                    stop_words_list=self.stop_words_list,
                    bad_words_list=self.bad_words_list,
                    output_cum_log_probs=(self.output_cum_log_probs_npy != None),
                    output_log_probs=(self.output_log_probs_npy != None),
                    lora_uids=self.lora_task_uids,
                    prompt_table=self.prompt_table_path,
                    prompt_tasks=self.prompt_tasks,
                    medusa_choices=self.medusa_choices)
                torch.cuda.synchronize()

            if streaming:

                previous_text = ""
                for curr_outputs in throttle_generator(outputs,
                                                    streaming_interval):
                    if self.runtime_rank == 0:
                        output_ids = curr_outputs['output_ids']
                        sequence_lengths = curr_outputs['sequence_lengths']
                        cum_log_probs = None
                        log_probs = None
                        if self.output_cum_log_probs_npy != None:
                            cum_log_probs = outputs['cum_log_probs']
                        if self.output_log_probs_npy != None:
                            log_probs = outputs['log_probs']

                        output_text = self.print_output(
                            self.tokenizer,
                            output_ids,
                            input_lengths,
                            sequence_lengths,
                            output_csv=self.output_csv,
                            output_npy=self.output_npy,
                            cum_log_probs=cum_log_probs,
                            log_probs=log_probs,
                            output_cum_log_probs_npy=self.output_cum_log_probs_npy,
                            output_log_probs_npy=self.output_log_probs_npy)
                        

                        torch.cuda.synchronize()
                        if output_text.endswith("</s>"):
                            output_text = output_text[:-4]
                        pre_token_len = len(previous_text)
                        new_text = output_text[pre_token_len:]  # Get only the new text

                        print(new_text, end='')
                        
                        """print("********************new new*********************")
                        print(new_text,end='')
                        print("*******************previous previous**********************")
                        #print(previous_text)
                        print("********************outputs outputs**********************")
                        #print(output_text)
                        print("*********************************************")"""

                        previous_text = output_text  # Update the previously yielded text after yielding
            
            else:
                if self.runtime_rank == 0:
                    output_ids = outputs['output_ids']
                    sequence_lengths = outputs['sequence_lengths']
                    context_logits = None
                    generation_logits = None
                    cum_log_probs = None
                    log_probs = None
                    if self.runner.gather_context_logits:
                        context_logits = outputs['context_logits']
                    if self.runner.gather_generation_logits:
                        generation_logits = outputs['generation_logits']
                    if self.output_cum_log_probs_npy != None:
                        cum_log_probs = outputs['cum_log_probs']
                    if self.output_log_probs_npy != None:
                        log_probs = outputs['log_probs']
                    output_text = self.print_output(self.tokenizer,
                                output_ids,
                                input_lengths,
                                sequence_lengths,
                                output_csv=self.output_csv,
                                output_npy=self.output_npy,
                                context_logits=context_logits,
                                generation_logits=generation_logits,
                                output_logits_npy=self.output_logits_npy,
                                cum_log_probs=cum_log_probs,
                                log_probs=log_probs,
                                output_cum_log_probs_npy=self.output_cum_log_probs_npy,
                                output_log_probs_npy=self.output_log_probs_npy)
                    print(output_text)
                    
    def run_profiler(self,input_text):

        self.input_text = [input_text]

        """self.input_file = None
        self.add_special_tokens = None
        self.max_input_length = None
        self.num_prepend_vtokens = None"""

        
        batch_input_ids = self.parse_input(tokenizer=self.tokenizer,
                                  input_text=self.input_text,
                                  prompt_template=self.prompt_template,
                                  input_file=None,
                                  add_special_tokens=self.add_special_tokens,
                                  max_input_length=self.max_input_length,
                                  pad_id=self.pad_id,
                                  num_prepend_vtokens=self.num_prepend_vtokens,
                                  model_name=self.model_name,
                                  model_version=self.model_version)
        ite = 10
        # warmup
        for _ in range(ite):
            with torch.no_grad():
                outputs = self.runner.generate(
                    batch_input_ids,
                    max_new_tokens=self.max_output_len,
                    max_attention_window_size=self.max_attention_window_size,
                    end_id=self.end_id,
                    pad_id=self.pad_id,
                    temperature=self.temperature,
                    top_k=1,
                    top_p=0.0,
                    num_beams=self.num_beams,
                    length_penalty=self.length_penalty,
                    early_stopping=1,
                    repetition_penalty=self.repetition_penalty,
                    presence_penalty=self.presence_penalty,
                    frequency_penalty=self.frequency_penalty,
                    stop_words_list=self.stop_words_list,
                    bad_words_list=self.bad_words_list,
                    lora_uids=self.lora_task_uids,
                    prompt_table=self.prompt_table_path,
                    prompt_tasks=self.prompt_tasks,
                    streaming=False,
                    output_sequence_lengths=True,
                    return_dict=True)
                torch.cuda.synchronize()

        tensorrt_llm.profiler.start("tmp")
        for _ in range(ite):
            with torch.no_grad():
                outputs = self.runner.generate(
                    batch_input_ids,
                    max_new_tokens=self.max_output_len,
                    max_attention_window_size=self.max_attention_window_size,
                    end_id=self.end_id,
                    pad_id=self.pad_id,
                    temperature=self.temperature,
                    top_k=1,
                    top_p=0.0,
                    num_beams=self.num_beams,
                    length_penalty=self.length_penalty,
                    early_stopping=1,
                    repetition_penalty=self.repetition_penalty,
                    presence_penalty=self.presence_penalty,
                    frequency_penalty=self.frequency_penalty,
                    stop_words_list=self.stop_words_list,
                    bad_words_list=self.bad_words_list,
                    lora_uids=self.lora_task_uids,
                    prompt_table=self.prompt_table_path,
                    prompt_tasks=self.prompt_tasks,
                    streaming=False,
                    output_sequence_lengths=True,
                    return_dict=True)
                torch.cuda.synchronize()
        
        tensorrt_llm.profiler.stop("tmp")

        print(
            f"batch_size: {len(batch_input_ids)}, avg latency of {ite} iterations: : {tensorrt_llm.profiler.elapsed_time_in_sec('tmp') / ite} sec"
        )

    def unload_model(self):
        if self.runner is not None:
            del self.runner
        # Step 3: Additional cleanup if needed
        torch.cuda.empty_cache()
        gc.collect()
        self.runner = None

    def reload_model(self):
        if self.runner is None:
            self.runner  = self.initialize_model_runner()
