#!/usr/bin/env nextflow
// QuantisedEncoder: train a matrix of model sizes x seeds (5k steps each), then PTQ-sweep
// each checkpoint. Thin orchestration only -- all model logic lives in bin/{train,quantise}.py.
nextflow.enable.dsl = 2


process TRAIN {
    tag { id }
    publishDir path: { "${params.outdir}/${id}" }, mode: 'copy'

    input:
    tuple val(id), val(size), val(seed), path(data)

    output:
    tuple val(id), val(size), val(seed), path("best_seed${seed}.pth"), emit: ckpt
    path "results_seed${seed}.json"
    path "wandb", optional: true                         // offline W&B run dir -> sync from login node later

    script:
    """
    export WANDB_MODE=offline
    python ${projectDir}/bin/train.py \
        --data_file ${data} --out_dir . \
        --d_model ${size.d} --num_heads ${size.h} --num_layers ${size.l} --d_ff ${size.ff} \
        --seed ${seed} --max_steps ${params.max_steps} --eval_every ${params.eval_every} \
        --run_name ${id} --wandb_group ${size.tag}
    """
}


process QUANTISE {
    tag { id }
    publishDir path: { "${params.outdir}/${id}" }, mode: 'copy'

    input:
    tuple val(id), val(size), val(seed), path(ckpt), path(data)

    output:
    path "quant_results.json"
    path "wandb", optional: true

    script:
    """
    export WANDB_MODE=offline
    export WANDB_DATA_DIR=\$PWD      # \$HOME is read-only on compute nodes; wandb.Table stages an artifact here
    python ${projectDir}/bin/quantise.py ${ckpt} --data_file ${data} --out_dir . \
        --run_name ${id}_quant --wandb_group ${size.tag}
    """
}


workflow {
    data = file(params.data, checkIfExists: true)

    experiments = Channel.fromList(params.sizes)
        .combine(Channel.fromList(params.seeds))                        // size x seed
        .map { size, seed -> tuple("${size.tag}_s${seed}", size, seed, data) }

    TRAIN(experiments)
    QUANTISE(TRAIN.out.ckpt.map { id, size, seed, ckpt -> tuple(id, size, seed, ckpt, data) })
}
