. ./cmd.sh
. ./path.sh

steps/decode.sh --nj 5 --cmd "run.pl" exp/tri1/graph_nosp_tgsmall \
	  data/decode_nosp_tgsmall_dev_clean_2 exp/tri1/decode_nosp_tgsmall_dev_clean_2

lattice-copy --write-compact=false ark:data/decode_nosp_tgsmall_dev_clean_2/lat.1.gz ark,t:lat1.txt
lattice-copy --write-compact=true ark:data/decode_nosp_tgsmall_dev_clean_2/lat1.gz ark,t:compact_lat1.txt

./utils/int2sym.pl -f 3 ./data/lang/words.txt ./exp/tri1/decode_nosp_tgsmall_dev_clean_2/lat1.txt >./exp/tri1/decode_nosp_tgsmall_dev_clean_2/ali1
./utils/int2sym.pl -f 3 ./data/lang/words.txt ./exp/tri1/decode_nosp_tgsmall_dev_clean_2/compact_lat1.txt >./exp/tri1/decode_nosp_tgsmall_dev_clean_2/ali1_cop

cat ali1|head -n 185655|tail -n 2956 > lattice
cat ali1_cop|head -n 4085|tail -n 68 > compactlattice
