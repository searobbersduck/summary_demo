# !/usr/bin/env python3

import os
import fire

import numpy as np
import json
import random

from glob import glob

'invoke:' \
'python data.py convert_file0 ./file_0 ./file_0_0.txt'
'spm_train --input=./file_0_0.txt --model_prefix=xxx --vocab_size=6000 --character_coverage=0.9995 --model_type=bpe'
'spm_encode --model=./xxx.model --output_format=piece < ./file_0_0.txt > ./file_0_0_seg.txt'
'~/beast/code/work/idmg/bazel-bin/third_party/word2vec/word2vec -train file_0_0_seg.txt -save-vocab word2vec_vocab.txt'
'~/beast/code/work/idmg/bazel-bin/third_party/word2vec/word2vec -train ./file_0_0_seg.txt -output xxx.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -iter 3 -cbow 0'
def convert_file0(infile, outfile):
    cont_list = []
    cont_sub_list = []
    with open(infile, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or line == '':
                cont_sub_list.append(cont_sub_list)
                cont_sub_list = []
                continue
            ss = line.split('\t')
            if len(ss) == 3:
                cont_list.append(ss[2])
            else:
                cont_list.append(line)
    with open(outfile, 'w', encoding='utf8') as f:
        f.write('\n'.join(cont_list))

global_keys_to_deleted = [
            'updatedAt',
            'updatedBy',
            'resumeId',
            'id',
            'createdBy',
            'createdAt',
            'underlingNumber', #???
            'mobile',
            'email',
            'age',
            'evaluate', # 推荐报告
            'avatar',
        ]

global_begin_line_per_resume = 'resumebegbegbegbegbegbegbegbegbegbeg'
global_end_line_per_resume = 'resumeendendendendendendendend'
global_begin_line_per_summary = 'summarybegbegbegbegbegbegbegbegbegbeg'
global_end_line_per_summary = 'summaryendendendendendendendend'

# global_begin_line_per_resume_seg = '▁re s ume be g be g be g be g be g be g be g be g be g be g'
# global_end_line_per_resume_seg = '▁re s ume en den den den den den den den d'
# global_begin_line_per_summary_seg = '▁s um m ary be g be g be g be g be g be g be g be g be g be g'
# global_end_line_per_summary_seg = '▁s um m ary en den den den den den den den d'

global_begin_line_per_resume_seg = 're sum e be gb eg be gb eg be gb eg be gb eg be gb eg'
global_end_line_per_resume_seg = 're sum e end end end end end end end end'
global_begin_line_per_summary_seg = 'summ ary be gb eg be gb eg be gb eg be gb eg be gb eg'
global_end_line_per_summary_seg = 'summ ary end end end end end end end end'

def load_keyspair_eng_chin(infile, vocab):
    with open(infile, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or line == '':
                continue
            ss = line.split('\t')
            if len(ss) != 3:
                continue
            eng_key = ss[0][:-1]
            chin_key = ss[2]
            vocab[eng_key] = chin_key

def convert_single_json_to_str(str):
    json_str = json.loads(str)
    summary = json_str['evaluate']
    resume_str = json_str['resumeJson']
    resume_json = json.loads(resume_str)


def loadvocab(infile, vocab):
    vocab_size = None
    dim = None
    # vocab = {}
    cnt = 0
    vec_list = []
    with open(infile, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or line == '':
                continue
            ss = line.split(' ')
            if len(ss) == 2:
                vocab_size = int(ss[0])
                dim = int(ss[1])
            else:
                vocab[ss[0]] = cnt
                cnt += 1
                vec_list.append(ss[1:])
    # Add </s> <unk>
    vocab_mat = np.array(vec_list)
    eos_vec = np.random.rand(1, dim)
    unk_vec = np.random.rand(1, dim)
    vocab_mat = np.concatenate([vocab_mat, eos_vec], axis=0)
    vocab_mat = np.concatenate([vocab_mat, unk_vec], axis=0)
    vocab['</s>'] = cnt
    cnt += 1
    vocab['<unk>'] = cnt
    return vocab_mat

def test_load():
    vocab = {}
    loadvocab('./xxx.txt', vocab)

class Vocabulary(object):
    def __init__(self, filename):
        '''
        :param filename: the file includes word & vector
        '''
        self._id_to_word = []
        self._word_to_id = {}
        self._bos = 1
        self._eos = 2
        self._unk = 3
        # self._strange_list = []
        vec_list = []
        vocab_size = None
        emb_dim = None
        idx = 4 # 1: bos, 2:eos, 3:unk
        vec_eos = None
        with open(filename, 'r', encoding='utf8') as f:
            for line1 in f.readlines():
                line = line1.strip()
                if line is None or line == '':
                    continue
                ss = line.split(' ')
                if len(ss) == 2:
                    vocab_size = int(ss[0])
                    emb_dim = int(ss[1])
                    continue
                if ss[0] == '</s>':
                    vec_eos = ss[1:]
                    continue
                if ss[0] == '<s>':
                    continue
                if len(ss) != 181:
                    # self._strange_list.append(line1[0])
                    continue
                self._word_to_id[ss[0]] = idx
                idx += 1
                vec_list.append(ss[1:])
        assert len(vec_list) == len(self._word_to_id.keys())
        emb_mat = np.array(vec_list)
        empty_vec = np.zeros((1, emb_dim))
        bos_vec = np.random.rand(1, emb_dim)
        eos_vec = np.expand_dims(np.array(vec_eos), axis=0)
        unk_vec = np.random.rand(1, emb_dim)
        self._emb_mat = np.concatenate([empty_vec, bos_vec, eos_vec, unk_vec, emb_mat], axis=0)
        self._word_to_id['<s>'] = 1
        self._word_to_id['</s>'] = 2
        self._word_to_id['unk'] = 3
        sorted_vocb = sorted(self._word_to_id.items(), key=lambda x:x[1], reverse=False)
        self._id_to_word.append('')
        for key, val in sorted_vocb:
            self._id_to_word.append(key)

    @property
    def bos(self):
        return self._bos

    @property
    def eos(self):
        return self._eos

    @property
    def unk(self):
        return self._unk

    @property
    def emb(self):
        return self._emb_mat

    @property
    def size(self):
        return len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        else:
            return self.unk

    def id_to_word(self, id):
        if id >= len(self._id_to_word):
            return 'unk'
        return self._id_to_word[id]

    def encode(self, sentence, reverse=False):
        sentence1 = sentence.split(' ')
        if sentence1[-1] == '':
            sentence1 = sentence1[:-1]
        word_ids = [self.word_to_id(word) for word in sentence1]
        if reverse:
            return np.array([self.eos] + word_ids + [self.bos], dtype=np.int32)
        else:
            return np.array([self.bos] + word_ids + [self.eos], dtype=np.int32)

    def decode(self, ids):
        pos = len(ids)
        for i, id in enumerate(ids):
            if id == 2:
                pos = i
                break
        ids1 = ids[:pos+1]
        return ' '.join([self.id_to_word(id) for id in ids1])

def read_summary_json(infile):
    with open(infile, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or line == '':
                continue
            cont = json.loads(line)
            summary = cont['evaluate']
            resume = cont['resumeJson']
            resume_json = json.loads(resume)
            print(summary)
            print('\n\n\n')
        print('hello world!')

def test_read_summary_json():
    infile = './filter_resume_evaluate.txt'
    read_summary_json(infile)

def read_all_keys(infile):
    keys_list = []
    keys_dict = {}
    def func_x(key, vocab):
        if key in vocab:
            vocab[key] += 1
        else:
            vocab[key] = 1
    with open(infile, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or line == '':
                continue
            all_json = json.loads(line)
            resumeJson = all_json['resumeJson']
            # resume
            resume_json = json.loads(resumeJson)
            # basic
            if 'basic' in resume_json:
                basic_json = resume_json['basic']
                for key, val in basic_json.items():
                    func_x(key, keys_dict)
            # edu
            if 'resumeEducations' in resume_json:
                edu_json = resume_json['resumeEducations']
                for edu in edu_json:
                    for key, val in edu.items():
                        func_x(key, keys_dict)
            # expr
            if 'resumeExperiences' in resume_json:
                expr_json = resume_json['resumeExperiences']
                for expr in expr_json:
                    for key, val in expr.items():
                        func_x(key, keys_dict)
            # proj
            if 'resumeProjects' in resume_json:
                proj_json = resume_json['resumeProjects']
                for proj in proj_json:
                    for key, val in proj.items():
                        func_x(key, keys_dict)
    logger = []
    for key, val in keys_dict.items():
        logger.append('{}:\t{}'.format(key, val))
        print('{}:\t{}'.format(key, val))
    print('====> keys num:\t{}'.format(len(keys_dict.keys())))
    logger.append('====> keys num:\t{}'.format(len(keys_dict.keys())))
    outdir = os.path.dirname(infile)
    outfile = os.path.join(outdir, 'all_keys.txt')
    with open(outfile, 'w', encoding='utf8') as f:
        f.write('\n'.join(logger))

def test_read_all_keys():
    # infile = './filter_resume_evaluate.txt'
    infile = '/Users/higgs/beast/tmp/summary/resume_evaluate.txt'
    read_all_keys(infile)

class ResumeJsonExtractor(object):
    def __init__(self,
                 delete_redundancy=True):
        self.name = 'ResumeJsonExtractor'
        self.summary = None
        self.resume = None
        self.basicinfo = None
        self.edu_list = None
        self.workexp_list = None
        self.proj_list = None
        self.para_list = ['basic', 'resumeEducations',
                          'resumeExperiences', 'resumeProjects']
        self.load_func_list = [
            self.loadBasicJson,
            self.loadEduJson,
            self.loadWorkJson,
            self.loadProjJson
        ]
        self.load_fun_with_key_list = [
            self.loadBasicJsonWithKey,
            self.loadEduJsonWithKey,
            self.loadWorkJsonWithKey,
            self.loadProjJsonWithKey
        ]
        self.keys_should_deleted = [
            'updatedAt',
            'updatedBy',
            'resumeId',
            'id',
            'createdBy',
            'createdAt',
            'underlingNumber', #???
            'mobile',
            'email',
            'age',
            'evaluate', # 推荐报告
            'avatar',
        ]
        self.delete_redundancy = delete_redundancy

    def isKeyJumped(self, key):
        if not self.delete_redundancy:
            return False
        # if key in self.keys_should_deleted:
        if key in global_keys_to_deleted:
            return True
        return False

    def extract(self, json_str):
        contents = json.loads(json_str)
        self.summary = contents['evaluate']
        resume_str = contents['resumeJson']
        resume_json = json.loads(resume_str)
        resume_list = self.loadJson(resume_json)
        self.resume = '\n'.join(resume_list)
        return self.summary, self.resume

    def loadEduJson(self, edus_json):
        '''
        :param edu_json:
        :return: edu_list
        '''
        edu_list = []
        for edu_json in edus_json:
            edu_sub_list = []
            for key, val in edu_json.items():
                if self.isKeyJumped(key):
                    continue
                edu_sub_list.append(str(val))
            edu_sub_str = '\n'.join(edu_sub_list)
            edu_list.append(edu_sub_str)
        return edu_list

    def loadWorkJson(self, works_json):
        return self.loadEduJson(works_json)

    def loadProjJson(self, projs_json):
        return self.loadEduJson(projs_json)

    def loadBasicJson(self, basic_json):
        basic_list = []
        for key, val in basic_json.items():
            if self.isKeyJumped(key):
                continue
            basic_list.append(str(val))
        return basic_list

    def loadEduJsonWithKey(self, edus_json, vocab_eng_chin=None):
        edu_list = []
        for edu_json in edus_json:
            edu_sub_list = []
            for key, val in edu_json.items():
                if self.isKeyJumped(key):
                    continue
                if vocab_eng_chin is not None:
                    if key in vocab_eng_chin:
                        key = vocab_eng_chin[key]
                edu_sub_list.append('{}:{}'.format(key, val))
            edu_sub_str = '\n'.join(edu_sub_list)
            edu_list.append(edu_sub_str)
        return edu_list

    def loadWorkJsonWithKey(self, works_json, vocab_eng_chin=None):
        return self.loadEduJsonWithKey(works_json, vocab_eng_chin)

    def loadProjJsonWithKey(self, projs_json, vocab_eng_chin=None):
        return self.loadEduJsonWithKey(projs_json, vocab_eng_chin)

    def loadBasicJsonWithKey(self, basic_json, vocab_eng_chin=None):
        basic_list = []
        for key, val in basic_json.items():
            if self.isKeyJumped(key):
                continue
            if vocab_eng_chin is not None:
                if key in vocab_eng_chin:
                    key = vocab_eng_chin[key]
            basic_list.append('{}:{}'.format(key, val))
        return basic_list

    def loadJson(self, resume_json, vocab_eng_chin=None,
                 is_shuffle_list=[False, False, False, False],
                 is_with_key_list=[True, True, True, True]):
        '''
        load json
        :param is_shuffle_list: a bool list of length of 4, stands for whether
        shuffle on [basic, edu, work, proj]
        :param is_with_key_list: a bool list of length of 4, stand for whether
        extract info with key on [basic, edu, work, proj]
        :return:
        '''
        resume_cont_list = []
        resume_cont_list_all = []
        for i, is_with_key in enumerate(is_with_key_list):
            if not self.para_list[i] in resume_json:
                continue
            if is_with_key:
                resume_cont_list.append(self.load_fun_with_key_list[i](resume_json[self.para_list[i]], vocab_eng_chin))
            else:
                resume_cont_list.append(self.load_func_list[i](resume_json[self.para_list[i]]))
        # todo: is shuffle
        for l in resume_cont_list:
            resume_cont_list_all += l
        return resume_cont_list_all

def test_ResumeJsonExtractor():
    infile = './filter_resume_evaluate.txt'
    json_str = None
    with open(infile, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or len(line) == 0:
                continue
            if json_str is not None:
                continue
            line_json = json.loads(line)
            if len(line_json['evaluate']) < 20:
                continue
            json_str = line
    if json_str is not None:
        extractor = ResumeJsonExtractor()
        summary, resume = extractor.extract(json_str)
        print('====> summary:')
        print(summary)
        print('\n\n\n\n')
        print('====> resume:')
        print(resume)
        print('\n\n\n\n')
        print('====> summary word number is: ')
        print(len(summary))
        print('====> resume word number is: ')
        print(len(resume))

class ResumeTxtExtractor(object):
    def __init__(self):
        self.name = 'ResumeTxtExtractor'

    def extract(self, infile):
        summaries = []
        resumes = []
        summary = ''
        resume = []
        summary_flag = False
        resume_flag = False
        with open(infile, 'r', encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or line == '':
                    continue
                if line == global_begin_line_per_summary_seg:
                    summary_flag = True
                    resume_flag = False
                    summary = ''
                    continue
                if line == global_end_line_per_summary_seg:
                    summary_flag = False
                    resume_flag = False
                    summaries.append(summary)
                    continue
                if line == global_begin_line_per_resume_seg:
                    summary_flag = False
                    resume_flag = True
                    resume = []
                    continue
                if line == global_end_line_per_resume_seg:
                    summary_flag = False
                    resume_flag = False
                    resumes.append(resume)
                    continue
                if summary_flag:
                    summary += line + ' '
                if resume_flag:
                    resume.append(line)
        assert len(resumes) == len(summaries)
        return summaries, resumes

class DemoExtractor():
    def __init__(self, infile):
        self.name = 'DemoExtractor'
        self.infile = infile
        self.all_patches = []
        with open(infile, 'r', encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                self.all_patches.append(line)
        self.choose_patches = self.all_patches

    def gen_one_item(self):
        resume_line_max_num = 100
        summary_line_max_num = 5
        resume_line_num = random.randint(30, resume_line_max_num)
        summary_line_num = random.randint(3, summary_line_max_num)
        random.shuffle(self.choose_patches)
        resume_list = self.choose_patches[:resume_line_num]
        summary_list = np.random.choice(resume_list, summary_line_num)
        summary = '\n'.join(summary_list)
        # resume = '\n'.join(resume_list)
        # summary = summary_list
        resume = resume_list
        return summary, resume

    def extract(self, infile):
        resumes = []
        summaries = []
        for i in range(100):
            summary, resume = self.gen_one_item()
            summaries.append(summary)
            resumes.append(resume)
        return summaries, resumes

def convert_json2txt_for_sentencepiece(infile, delete_redundancy=False):
    vocab_eng_chin = {}
    extractor = ResumeJsonExtractor(delete_redundancy=delete_redundancy)
    load_keyspair_eng_chin('/Users/higgs/beast/tmp/summary/all_keys_chin.txt', vocab_eng_chin)
    outdir = os.path.dirname(infile)
    outfile = os.path.join(outdir, 'resume_evaluate_all.txt')
    with open(infile, 'r', encoding='utf8') as f:
        with open(outfile, 'w', encoding='utf8') as fout:
            for line in f.readlines():
                line = line.strip()
                if line is None or line == '':
                    continue
                all_json = json.loads(line)
                summary = all_json['evaluate']
                if len(summary) < 10:
                    continue
                resume_str = all_json['resumeJson']
                resuem_json = json.loads(resume_str)
                resume = extractor.loadJson(resuem_json, vocab_eng_chin=vocab_eng_chin)
                fout.write(summary)
                fout.write('\n')
                fout.write('\n'.join(resume))
                fout.write('\n')

def test_convert_json2txt_for_sentencepiece():
    infile = '/Users/higgs/beast/tmp/summary/resume_evaluate.txt'
    # infile = './filter_resume_evaluate.txt'
    convert_json2txt_for_sentencepiece(infile, True)

def convert_json2txt_for_seg(infile, delete_redundancy=True):
    num_lines = sum(1 for line in open(infile, 'r', encoding='utf8'))
    files_total_num = 200
    num_per_file = (num_lines + files_total_num -1) // files_total_num
    vocab_eng_chin = {}
    extractor = ResumeJsonExtractor(delete_redundancy=delete_redundancy)
    load_keyspair_eng_chin('/Users/higgs/beast/tmp/summary/all_keys_chin.txt', vocab_eng_chin)
    outdir = os.path.dirname(infile)
    outdir = os.path.join(outdir, 'for_seg')
    os.makedirs(outdir, exist_ok=True)
    # outfile = os.path.join(outdir, 'resume_evaluate_all.txt')
    cnt_per_file = 0
    list_per_file = []
    num_file = 0
    with open(infile, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or line == '':
                continue
            all_json = json.loads(line)
            summary = all_json['evaluate']
            if len(summary) < 10:
                continue
            resume_str = all_json['resumeJson']
            resuem_json = json.loads(resume_str)
            resume = extractor.loadJson(resuem_json, vocab_eng_chin=vocab_eng_chin)
            resume = '\n'.join(resume)
            cnt_per_file += 1
            list_per_file.append(global_begin_line_per_summary)
            list_per_file.append(summary)
            list_per_file.append(global_end_line_per_summary)
            list_per_file.append(global_begin_line_per_resume)
            list_per_file.append(resume)
            list_per_file.append(global_end_line_per_resume)
            if cnt_per_file % num_per_file == 0:
                cnt_per_file = 0
                num_file += 1
                outfile = os.path.join(outdir, 'file_{}.txt'.format(num_file))
                with open(outfile, 'w', encoding='utf8') as fout:
                    fout.write('\n'.join(list_per_file))
                list_per_file = []


def test_convert_json2txt_for_seg():
    infile = '/Users/higgs/beast/tmp/summary/resume_evaluate.txt'
    convert_json2txt_for_seg(infile)

def _get_batch(generator, vocab, batch_size,
               resume_max_line, resume_max_tokens_per_line,
               summary_max_tokens_per_sent):
    while True:
        resumes_inputs = np.zeros(
            [batch_size, resume_max_line, resume_max_tokens_per_line], dtype=np.int32)
        summary_inputs = np.zeros(
            [batch_size, summary_max_tokens_per_sent], dtype=np.int32
        )
        for i in range(batch_size):
            summary, resume = next(generator)
            summary_tokens = vocab.encode(summary)
            cut_len = min(summary_max_tokens_per_sent, len(summary_tokens))
            summary_inputs[i, :cut_len] = summary_tokens[:cut_len]
            for j in range(min(len(resume), resume_max_line)):
                r1 = vocab.encode(resume[j])
                cut_len = min(resume_max_tokens_per_line, len(r1))
                resumes_inputs[i, j, :cut_len] = r1[:cut_len]
        yield summary_inputs, resumes_inputs

        # result = next(generator)
        # yield result


class ResumeSummaryDataset(object):
    def __init__(self, filepattern,
                 vocab,
                 extractor,
                 test=False,
                 shuffle_on_load=False):
        self._vocab = vocab
        self._extractor = extractor
        self._all_shards = glob(filepattern)
        print('Found {} shards at {}'.format(len(self._all_shards), filepattern))
        self._shards_to_choose = []
        self._test = test
        self._shuffle_on_load = shuffle_on_load
        self._infos = self._load_random_shard()

    def _choose_random_shard(self):
        if len(self._shards_to_choose) == 0:
            self._shards_to_choose = list(self._all_shards)
            random.shuffle(self._shards_to_choose)
        shard_name = self._shards_to_choose.pop()
        return shard_name

    def _load_shard(self, shard_name):
        print('load data from {}'.format(shard_name))
        resumes = []
        summarys = []
        # with open(shard_name, 'r', encoding='utf8') as f:
        #     for line in f.readlines():
        #         line = line.strip()
        #         if line is None or len(line) == 0:
        #             continue
        #         summary, resume = self._extractor.extract(line)
        #         resumes.append(resume)
        #         summarys.append(summary)
        summarys, resumes = self._extractor.extract(shard_name)
        if self._shuffle_on_load:
            ids = [i for i in range(len(summarys))]
            random.shuffle(ids)
            resumes = [resumes[i] for i in ids]
            summarys = [summarys[i] for i in ids]
        return list(zip(summarys, resumes))

    def _load_random_shard(self):
        if self._test:
            if len(self._all_shards) == 0:
                raise StopIteration
            else:
                shard_name = self._all_shards.pop()
        else:
            shard_name = self._choose_random_shard()
        infos = self._load_shard(shard_name)
        self._i = 0
        self._ninfos = len(infos)
        return infos

    def get_sentence(self):
        while True:
            if self._i == self._ninfos:
                self._infos = self._load_random_shard()
            ret = self._infos[self._i]
            self._i += 1
            yield ret

    def iter_batches(self, batch_size,
                     resume_line_max_num, resume_max_tokens_per_line,
                     summary_max_tokens_per_sent):
        for X in _get_batch(self.get_sentence(), self.vocab, batch_size,
                            resume_line_max_num,
                            resume_max_tokens_per_line,
                            summary_max_tokens_per_sent):
            yield X

    @property
    def vocab(self):
        return self._vocab


def test_ResumeSummaryDataset():
    extractor = DemoExtractor('./file_0_0_seg.txt')
    vocab = Vocabulary('./xxx.txt')
    ds = ResumeSummaryDataset('./file_0_0_*', vocab, extractor)
    resume_line_max_num = 300
    resume_max_tokens_per_line = 20
    summary_max_tokens_per_sent = 200
    for i in range(1000):
        X = ds.iter_batches(1, resume_line_max_num,
                            resume_max_tokens_per_line,
                            summary_max_tokens_per_sent)
        Y = next(X)
        print(Y)
        print('hello world!')
    print('hello world!')

def test_ResumeSummaryDataset_resume():
    extractor = ResumeTxtExtractor()
    # vocab = Vocabulary('./summary_data/xxx.txt')
    vocab = Vocabulary('./summary_data/v160k_big_string.txt')
    ds = ResumeSummaryDataset('./summary_data/seg/file_*', vocab, extractor)
    resume_line_max_num = 500
    resume_max_tokens_per_line = 50
    summary_max_tokens_per_sent = 500
    for i in range(1000):
        X = ds.iter_batches(1, resume_line_max_num,
                            resume_max_tokens_per_line,
                            summary_max_tokens_per_sent)
        Y = next(X)
        print(Y[1])
        print('hello world!')
    print('hello world!')

def generate_seg_script(indir, model):
    # spm_encode --model =./ xxx.model - -output_format = piece <./ file_0_0.txt >./ file_0_0_seg.txt
    rootdir = os.path.dirname(indir)
    script_file = os.path.join(rootdir, 'gen_seg_resume.sh')
    outdir = os.path.join(rootdir, 'seg')
    os.makedirs(outdir, exist_ok=True)
    infiles = glob(os.path.join(indir, 'file_*'))
    cmd_list = []
    for infile in infiles:
        outfile = os.path.join(outdir, os.path.basename(infile))
        cmd = 'spm_encode --model={} --output_format=piece <{} >{}'.format(model, infile, outfile)
        cmd_list.append(cmd)
    with open(script_file, 'w') as f:
        f.write('\n'.join(cmd_list))

def merge_multi_seg_files_to_onefile():
    rootdir = './summary_data'
    outfile = os.path.join(rootdir, 'file_to_gen_word2vec.txt')
    indir = os.path.join(rootdir, 'seg')
    infiles = glob(os.path.join(indir, 'file_*'))
    with open(outfile, 'w', encoding='utf8') as fout:
        for infile in infiles:
            with open(infile, 'r', encoding='utf8') as f:
                for line in f.readlines():
                    fout.write(line)


'''
====> summary max tokens number is:	1941
====> resume max tokens number is:	1334
====> resume max line number is:	785

取：
====> summary max tokens number is:	500
====> resume max tokens number is:	50
====> resume max line number is:	500
'''
def stat_resume_linenum_and_tokens(infile):
    summaries = []
    resumes = []
    summary = ''
    resume = []
    summary_flag = False
    resume_flag = False
    max_len_summary = 0
    max_linenum_resume = 0
    max_len_resume = 0
    extract_id_list = []
    with open(infile, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or line == '':
                continue
            if line == global_begin_line_per_summary_seg:
                summary_flag = True
                resume_flag = False
                summary = ''
                continue
            if line == global_end_line_per_summary_seg:
                summary_flag = False
                resume_flag = False
                summaries.append(summary)
                ss = summary.split(' ')
                if len(ss) > max_len_summary:
                    extract_id_list.append(len(summaries)-1)
                    max_len_summary = len(ss)
                    # print(max_len_summary)
                    # print(summary)
                    # print('\n\n\n')
                continue
            if line == global_begin_line_per_resume_seg:
                summary_flag = False
                resume_flag = True
                resume = []
                continue
            if line == global_end_line_per_resume_seg:
                summary_flag = False
                resume_flag = False
                if len(resume) > max_linenum_resume:
                    max_linenum_resume = len(resume)
                    print(max_linenum_resume)
                    print(resume)
                    print('\n\n\n')
                resumes.append(resume)
                continue
            if summary_flag:
                summary += line
            if resume_flag:
                ss = line.split(' ')
                if len(ss) > max_len_resume:
                    max_len_resume = len(ss)
                    # print(max_len_resume)
                    # print(line)
                    # print('\n\n\n')
                resume.append(line)
    print('====> summary max tokens number is:\t{}'.format(max_len_summary))
    print('====> resume max tokens number is:\t{}'.format(max_len_resume))
    print('====> resume max line number is:\t{}'.format(max_linenum_resume))
    with open('./demo.txt', 'w', encoding='utf8') as f:
        for id in extract_id_list:
            f.write(summaries[id])
            f.write('\n')
            f.write('\n'.join(resumes[id]))
            f.write('\n\n\n\n\n')


'''
matching summary with rouge
'''
from rouge import Rouge
rouge = Rouge()
def matching_resume_summary(summary, resume, line_num=4):
    reference = summary
    selected_list = []
    selected_str = ''
    for i in range(line_num):
        scores_list = []
        for j in range(len(resume)):
            if j not in selected_list:
                hypothesis = selected_str + resume[j]
                scores = rouge.get_scores(hypothesis, reference)
                scores_list.append(scores[0]['rouge-1']['f'])
            else:
                scores_list.append(0.0)
        selected_num = np.argmax(scores_list)
        max_score = np.argmax(selected_num)
        selected_list.append(selected_num)
        selected_str += resume[selected_num] + '\n'
    return selected_str

def matching_resume_summary_by_rouge(infile):
    summaries = []
    resumes = []
    summary = ''
    resume = []
    summary_flag = False
    resume_flag = False
    max_len_summary = 0
    max_linenum_resume = 0
    max_len_resume = 0
    extract_id_list = []
    matchings = []
    with open(infile, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or line == '':
                continue
            if line == global_begin_line_per_summary_seg:
                summary_flag = True
                resume_flag = False
                summary = ''
                continue
            if line == global_end_line_per_summary_seg:
                summary_flag = False
                resume_flag = False
                # summaries.append(summary)
                continue
            if line == global_begin_line_per_resume_seg:
                summary_flag = False
                resume_flag = True
                resume = []
                continue
            if line == global_end_line_per_resume_seg:
                summary_flag = False
                resume_flag = False
                # resumes.append(resume)
                matching_str = matching_resume_summary(summary, resume)
                matching_str_out = '====> summary:\n' + summary + '\n====> resume:\n' + matching_str
                matchings.append(matching_str_out)
                if len(matchings) == 100:
                    break
                continue
            if summary_flag:
                summary += line
            if resume_flag:
                resume.append(line)
    with open('./demo.txt', 'w', encoding='utf8') as f:
        f.write('\n\n\n\n'.join(matchings))



def replace_all_strange_chars():
    extractor = ResumeTxtExtractor()
    vocab = Vocabulary('./summary_data/v160k_big_string.txt')
    print('hello world!')
    infile_pattern = './summary_data/seg/file_*'
    infile_list = glob('./summary_data/seg/file_*')
    outdir = './summary_data/seg_new'
    os.makedirs(outdir, exist_ok=True)
    for infile in infile_list:
        basename = os.path.basename(infile)
        outfile = os.path.join(outdir, basename)
        with open(infile, 'r', encoding='utf8') as fin:
            with open(outfile, 'w', encoding='utf8') as fout:
                data = fin.read()
                for c in vocab._strange_list:
                    data = data.replace(c, '')
                fout.write(data)

    # ds = ResumeSummaryDataset('./summary_data/seg/file_*', vocab, extractor)



if __name__ == '__main__':
    # fire.Fire()
    # test_read_summary_json()
    # test_ResumeJsonExtractor()
    # test_ResumeSummaryDataset()
    # test_read_all_keys()
    # test_convert_json2txt_for_sentencepiece()
    # test_convert_json2txt_for_seg()
    # generate_seg_script('/Users/higgs/beast/tmp/summary/for_seg', './xxx.model')
    # generate_seg_script('/Users/higgs/beast/tmp/summary1/for_seg', './big_160k_spm.model')
    # merge_multi_seg_files_to_onefile()
    # stat_resume_linenum_and_tokens('./summary_data/file_to_gen_word2vec.txt')
    test_ResumeSummaryDataset_resume()
    # matching_resume_summary_by_rouge('./summary_data/seg/file_29.txt')
    # replace_all_strange_chars()