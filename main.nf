#!/usr/bin/env nextflow
// QuantisedEncoder pipeline:  PREP (cluster_id fan-out) -> TRAIN -> gate(top1) -> {EVAL_PGYM, QUANTISE} -> REPORT.
// The point of the fan-out: test whether keeping homolog depth (looser clustering) lifts ProteinGym.
// Thin orchestration only -- all logic lives in bin/{downloads,train,quantise,proteingym,contacts,report}.py.
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
               pattern: "{best_*.pth,results_*.json,wandb}"   // don't re-publish the fasta we pass through

    input:
    tuple val(id), val(cid), val(size), val(seed), path(data), path(pgym)

    output:
    tuple val(id), val(cid), val(size), val(seed), path("best_${params.dataset}_s${seed}.pth"), path("results_${params.dataset}_s${seed}.json"), path(data), emit: ckpt
    path "wandb", optional: true

    script:
    def distill = params.distill_teacher ? "--distill_teacher ${params.distill_teacher} --distill_weight ${params.distill_weight}" : ""
    def panel   = params.dev_match ? "--pgym_panel ${pgym} --pgym_panel_match ${params.dev_match}" : ""   // select on held-out assays
    """
    python ${projectDir}/bin/train.py \
        --data_file ${data} --dataset ${params.dataset} --out_dir . \
        --d_model ${size.d} --num_heads ${size.h} --num_layers ${size.l} --d_ff ${size.ff} \
        --seed ${seed} --max_steps ${params.max_steps} --eval_every ${params.eval_every} --ckpt_every ${params.ckpt_every} \
        --run_name ${id} --wandb_group ${params.dataset}_${cid}_${size.tag} \
        ${distill} ${panel}
    """
}


process EVAL_PGYM {                                   // zero-shot ProteinGym (per-dataset eval set) + ESM2-8M baseline
    tag { id }
    publishDir path: { "${params.outdir}/${id}" }, mode: 'copy', pattern: "{proteingym_*.csv,row_*.json}"

    input:
    tuple val(id), val(cid), val(size), val(seed), path(ckpt), val(top1), val(blosum), val(blosum_null), path(pgym)

    output:
    path "row_${id}.json"

    script:
    def excl = params.dev_match ? "--exclude ${params.dev_match}" : ""   // keep report disjoint from the dev panel
    """
    python ${projectDir}/bin/proteingym.py ${ckpt} \
        --dms_dir ${pgym} --names fp --baseline ESM2_8M ${params.datasets[params.dataset].pgym_eval} ${excl} \
        --out proteingym_${id}.csv --summary_json summary.json
    python -c "import json; s=json.load(open('summary.json')); json.dump({'id':'${id}','dataset':'${params.dataset}','cluster_id':'${cid}','tag':'${size.tag}','seed':${seed},'mlm_top1':${top1},'blosum':${blosum},'blosum_null':${blosum_null},'pgym_fp':s.get('fp'),'pgym_esm2':s.get('ESM2_8M'),'wt_nll':s.get('fp_wtnll')}, open('row_${id}.json','w'), indent=2)"
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


process CONTACT_PROBE {                               // unsupervised attention-contact probe (Rao et al.) -- diagnostic, off by default
    tag { id }
    publishDir path: { "${params.outdir}/${id}" }, mode: 'copy', pattern: "contacts_*.json"

    input:
    tuple val(id), path(ckpt), path(pdb)

    output:
    path "contacts_${id}.json"

    script:
    """
    python ${projectDir}/bin/contacts.py ${ckpt} --pdb_dir ${pdb} --out contacts_${id}.json
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
    raw  = file(params.datasets[params.dataset].raw, checkIfExists: true)   // resolved here so --dataset overrides take effect
    pgym = file(params.pgym_dir, checkIfExists: true)

    PREP(Channel.fromList(params.cluster_ids).map { cid -> tuple(cid, raw) })

    experiments = PREP.out
        .combine(Channel.fromList(params.sizes))
        .combine(Channel.fromList(params.seeds))
        .map { cid, data, size, seed -> tuple("${params.dataset}_${cid}_${size.tag}_s${seed}", cid, size, seed, data, pgym) }   // id carries the dataset -> results/<dataset>_<cid>_<tag>_s<seed>/

    TRAIN(experiments)

    // gate: pull MLM top1 + BLOSUM (model & freq-null) from results json, keep only checkpoints above the bar
    gated = TRAIN.out.ckpt
        .map { id, cid, size, seed, ckpt, res, data ->
            def j = new groovy.json.JsonSlurper().parseText(res.text)
            tuple(id, cid, size, seed, ckpt, data, j.test.top1, j.blosum.spearman_offdiag, j.blosum_null)
        }
        .filter { it[6] >= params.gate_top1 }

    EVAL_PGYM(gated.map { id, cid, size, seed, ckpt, data, top1, blosum, blosum_null -> tuple(id, cid, size, seed, ckpt, top1, blosum, blosum_null, pgym) })
    QUANTISE(gated.map { id, cid, size, seed, ckpt, data, top1, blosum, blosum_null -> tuple(id, size, seed, ckpt, data) })
    if (params.contacts_dir) {                        // optional Tier-2 diagnostic; not a champion selector
        cpdb = file(params.contacts_dir, checkIfExists: true)
        CONTACT_PROBE(gated.map { id, cid, size, seed, ckpt, data, top1, blosum, blosum_null -> tuple(id, ckpt, cpdb) })
    }
    REPORT(EVAL_PGYM.out.collect())
}
