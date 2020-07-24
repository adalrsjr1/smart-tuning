#for dot in $(ls graph_*.dot); do dot $dot -Tpng -Gsize=25.6,16\! -Gdpi=300 > gif/output_$dot.png; done
rm -f gif/*.png
for dot in $(ls graph_*.dot); do dot $dot -Tpng -Gdpi=300 > gif/output_$dot.png; done
