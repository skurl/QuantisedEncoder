#!/usr/bin/env nextflow
// QuantisedEncoder pipeline:  PREP (cluster_id fan-out) -> TRAIN -> gate(top1) -> {EVAL_PGYM, QUANTISE} -> REPORT.
// The point of the fan-out: test whether keeping homolog depth (looser clustering) lifts ProteinGym.
// Thin orchestration only -- all logic lives in bin/{downloads,train,quantise,proteingym,report}.py.
nextflow.enable.dsl = 2


process PREP {                                        // one training set per clustering threshold
    tag { cid }
    publishDir path: { "${params.outdir}/data" }, mode: 'copy'

    input:
    tuple val(cid), path(raw)

    output:
    tuple val(cid), path("dataset_${cid}.fasta")

    script:
    if (cid == 'none')
        """
        cp ${raw} dataset_none.fasta                 # no clustering -> full homolog depth
        """
    else
        """
        mmseqs easy-linclust ${raw} clust tmp --min-seq-id ${cid} -c ${params.cluster_cov}
        mv clust_rep_seq.fasta dataset_${cid}.fasta
        """
}


process TRAIN {
    tag { id }
    publishDir path: { "${params.outdir}/${id}" }, mode: 'copy',
               pattern: "{best_seed*.pth,results_seed*.json,wandb}"   // don't re-publish the fasta we pass through

    input:
    tuple val(id), val(cid), val(size), val(seed), path(data)

    output:
    tuple val(id), val(cid), val(size), val(seed), path("best_seed${seed}.pth"), path("results_seed${seed}.json"), path(data), emit: ckpt
    path "wandb", optional: true

    script:
    """
    python ${projectDir}/bin/train.py \
        --data_file ${data} --out_dir . \
        --d_model ${size.d} --num_heads ${size.h} --num_layers ${size.l} --d_ff ${size.ff} \
        --seed ${seed} --max_steps ${params.max_steps} --eval_every ${params.eval_every} \
        --run_name ${id} --wandb_group ${cid}_${size.tag}
    """
}


process EVAL_PGYM {                                   // zero-shot ProteinGym (fungal subset) + ESM2-8M baseline
    tag { id }
    publishDir path: { "${params.outdir}/${id}" }, mode: 'copy', pattern: "{proteingym_*.csv,row_*.json}"

    input:
    tuple val(id), val(cid), val(size), val(seed), path(ckpt), val(top1), path(pgym)

    output:
    path "row_${id}.json"

    script:
    """
    python ${projectDir}/bin/proteingym.py ${ckpt} \
        --dms_dir ${pgym} --names fp --baseline ESM2_8M --match ${params.pgym_match} \
        --out proteingym_${id}.csv --summary_json summary.json
    python -c "import json; s=json.load(open('summary.json')); json.dump({'id':'${id}','cluster_id':'${cid}','tag':'${size.tag}','seed':${seed},'mlm_top1':${top1},'pgym_fp':s.get('fp'),'pgym_esm2':s.get('ESM2_8M')}, open('row_${id}.json','w'), indent=2)"
    """
}


process QUANTISE {
    tag { id }
    publishDir path: { "${params.outdir}/${id}" }, mode: 'copy', pattern: "{quant_results.json,wandb}"

    input:
    tuple val(id), val(size), val(seed), path(ckpt), path(data)

    output:
    path "quant_results.json"
    path "wandb", optional: true

    script:
    """
    python ${projectDir}/bin/quantise.py ${ckpt} --data_file ${data} --out_dir . \
        --run_name ${id}_quant --wandb_group ${size.tag}
    """
}


process REPORT {                                      // rank checkpoints, pick a champion
    publishDir path: { "${params.outdir}" }, mode: 'copy'

    input:
    path rows

    output:
    path "champion.json"
    path "leaderboard.csv"

    script:
    """
    python ${projectDir}/bin/report.py ${rows}
    """
}


workflow {
    raw  = file(params.raw,      checkIfExists: true)
    pgym = file(params.pgym_dir, checkIfExists: true)

    PREP(Channel.fromList(params.cluster_ids).map { cid -> tuple(cid, raw) })

    experiments = PREP.out
        .combine(Channel.fromList(params.sizes))
        .combine(Channel.fromList(params.seeds))
        .map { cid, data, size, seed -> tuple("${cid}_${size.tag}_s${seed}", cid, size, seed, data) }

    TRAIN(experiments)

    // gate: attach MLM test top1 from results json, keep only checkpoints above the bar
    gated = TRAIN.out.ckpt
        .map { id, cid, size, seed, ckpt, res, data ->
            def top1 = new groovy.json.JsonSlurper().parseText(res.text).test.top1
            tuple(id, cid, size, seed, ckpt, data, top1)
        }
        .filter { it[6] >= params.gate_top1 }

    EVAL_PGYM(gated.map { id, cid, size, seed, ckpt, data, top1 -> tuple(id, cid, size, seed, ckpt, top1, pgym) })
    QUANTISE(gated.map { id, cid, size, seed, ckpt, data, top1 -> tuple(id, size, seed, ckpt, data) })
    REPORT(EVAL_PGYM.out.collect())
}
