#!/bin/bash

# ----------- Configurations ------------
fast5_dir="fast5"
basecalled_dir="result/1-basecalled"
fastq_dir="result/1-fastq"
aligned_dir="result/2-aligned"
eventalign_dir="result/4-eventalign"
REF="transcriptome fasta"
flowcell="FLO-MIN106"
kit="SQK-RNA001"
threads=40

# ----------- Step 1: Basecalling ------------
mkdir -p "$basecalled_dir"
guppy_basecaller --flowcell "$flowcell" \
                 --kit "$kit" \
                 --num_callers "$threads" \
                 -i "$fast5_dir" \
                 -s "$basecalled_dir" \
                 -r --fast5_out

# ----------- Step 2: FASTQ Processing ------------
mkdir -p "$fastq_dir"
cat "$basecalled_dir"/pass/*.fastq > "$fastq_dir/output.fastq"

# Quality filtering and trimming
NanoFilt -q 0 --headcrop 5 --tailcrop 3 --readtype 1D < "$fastq_dir/output.fastq" \
  > "$fastq_dir/h5t3.fastq"

# Convert U to T (required by alignment tools)
awk '{ if (NR % 4 == 2) { gsub(/U/, "T", $1); print $1 } else print }' \
  "$fastq_dir/h5t3.fastq" > "$fastq_dir/U2T.fastq"

# ----------- Step 3: Alignment ------------
mkdir -p "$aligned_dir"
minimap2 -G200k --secondary=no -ax splice -uf -k14 -t "$threads" \
         "$REF" "$fastq_dir/U2T.fastq" > "$aligned_dir/raw.sam"

# Filter primary alignments (flag 0 or 16) and sort
samtools view -H "$aligned_dir/raw.sam" > "$aligned_dir/filtered.sam"
awk '{if(($2=="0")||($2=="16")) print $0}' "$aligned_dir/raw.sam" >> "$aligned_dir/filtered.sam"

samtools view -bS "$aligned_dir/filtered.sam" | samtools sort -@ "$threads" \
  -o "$aligned_dir/filtered.sorted.bam"
samtools index "$aligned_dir/filtered.sorted.bam"

# ----------- Step 4: Nanopolish Eventalign ------------
mkdir -p "$eventalign_dir"
nanopolish index -d "$fast5_dir" "$fastq_dir/U2T.fastq"

nanopolish eventalign --reads "$fastq_dir/U2T.fastq" \
                      --bam "$aligned_dir/filtered.sorted.bam" \
                      --genome "$REF" \
                      --samples --signal-index --scale-events \
                      --summary "$eventalign_dir/summary.txt" \
                      -t "$threads" \
                      > "$eventalign_dir/eventalign.txt"

echo "✅ Done! Final output: $eventalign_dir/eventalign.txt"
