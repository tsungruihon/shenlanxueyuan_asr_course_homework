. ./cmd.sh
. ./path.sh

gunzip -c data/local/lm/lm_tglarge.arpa.gz | \
	  arpa2fst --disambig-symbol=#0 \
	               --read-symbol-table=/home/jovyan/data2/peter/kaldi/egs/mini_librispeech/s5/data/lang_nosp_test_tglarge/words.txt - data/lang_nosp_test_tglarge/G.fst
