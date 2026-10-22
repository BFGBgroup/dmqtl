#!/bin/bash
# Sample the real memory of a process tree every few seconds until its root exits.
# usage: mem_sampler.sh <root pid> <out tsv> [interval s]
# Columns (kB):
#   anon_pss   anonymous memory with pages shared after fork split between the processes = real heap total
#   file_pss   memory-mapped files (e.g. the genotype store), also split
#   rss_sum    sum of RSS over processes, what Slurm's MaxRSS adds up (counts shared pages once per process)
#   main_priv  private anonymous memory of the root (start the python process itself as the root)
#   child_priv largest private anonymous memory of any descendant (copy-on-write copies + own allocations)
root=$1; out=$2; iv=${3:-5}
printf "t_s\tnproc\tanon_pss\tfile_pss\trss_sum\tmain_priv\tchild_priv\n" > "$out"
t0=$(date +%s)
desc() { local p=$1; echo "$p"; for c in $(pgrep -P "$p"); do desc "$c"; done; }
while kill -0 "$root" 2>/dev/null; do
  A=0; F=0; R=0; MP=0; CP=0; n=0
  for p in $(desc "$root"); do
    read a f r ap < <(awk '/^[0-9a-f]+-[0-9a-f]+ /{file=($6 ~ /^\//)} /^Pss:/{if(file)f+=$2; else a+=$2}
                          /^Rss:/{r+=$2} /^Private_(Clean|Dirty):/{if(!file)ap+=$2}
                          END{print a+0, f+0, r+0, ap+0}' /proc/"$p"/smaps 2>/dev/null)
    [ -z "$a" ] && continue
    n=$((n+1)); A=$((A+a)); F=$((F+f)); R=$((R+r))
    if [ "$p" = "$root" ]; then MP=$ap; elif [ "$ap" -gt "$CP" ]; then CP=$ap; fi
  done
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $(( $(date +%s) - t0 )) $n $A $F $R $MP $CP >> "$out"
  sleep "$iv"
done
awk -F'\t' 'NR>1{for(i=3;i<=7;i++) if($i>m[i]) m[i]=$i} END{printf "MEMPEAK anon_pss=%.2f GB file_pss=%.2f GB rss_sum=%.2f GB main_priv=%.2f GB child_priv=%.2f GB\n", m[3]/1048576, m[4]/1048576, m[5]/1048576, m[6]/1048576, m[7]/1048576}' "$out"
