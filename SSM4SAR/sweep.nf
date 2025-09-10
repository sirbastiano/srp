#!/usr/bin/env nextflow

/*
 * Nextflow workflow for SSM4SAR hyperparameter sweeps
 * 
 * Usage:
 *   nextflow run sweep.nf --outdir results/sweep_$(date +%Y%m%d)
 */

nextflow.enable.dsl = 2

// Define parameter space for sweep
params.learning_rates = [0.001, 0.005, 0.01, 0.02]
params.batch_sizes = [8, 16, 32]
params.num_layers = [2, 4, 6, 8]
params.hidden_sizes = [8, 16, 32, 64]
params.activation_functions = ['relu', 'gelu', 'leakyrelu']
params.ssim_proportions = [0.3, 0.5, 0.7]
params.weight_decays = [0.001, 0.01, 0.1]

// Fixed parameters
params.epochs = 15
params.val_batch_size = 10
params.gpu_no = 1

// Directories
params.train_dir = '/Data_large/marine/PythonProjects/SAR/sarpyx/SSM4SAR/maya4_data/training'
params.val_dir = '/Data_large/marine/PythonProjects/SAR/sarpyx/SSM4SAR/maya4_data/validation'
params.outdir = 'results'

log.info """
SSM4SAR Hyperparameter Sweep
============================
Learning rates    : ${params.learning_rates}
Batch sizes       : ${params.batch_sizes}
Number of layers  : ${params.num_layers}
Hidden sizes      : ${params.hidden_sizes}
Activation funcs  : ${params.activation_functions}
SSIM proportions  : ${params.ssim_proportions}
Weight decays     : ${params.weight_decays}
Output directory  : ${params.outdir}
"""

/*
 * Generate parameter combinations
 */
process generateParams {
    output:
    path 'param_combinations.json'

    script:
    """
    #!/usr/bin/env python3
    import json
    import itertools

    # Define parameter grid
    param_grid = {
        'learning_rate': ${params.learning_rates},
        'batch_size': ${params.batch_sizes},
        'num_layers': ${params.num_layers},
        'hidden_state_size': ${params.hidden_sizes},
        'act_fun': ${params.activation_functions},
        'ssim': ${params.ssim_proportions},
        'weight_decay': ${params.weight_decays}
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))
    
    param_sets = []
    for i, combo in enumerate(combinations):
        param_dict = dict(zip(keys, combo))
        param_dict['experiment_id'] = f"sweep_exp_{i:04d}"
        param_sets.append(param_dict)
    
    # Save to JSON
    with open('param_combinations.json', 'w') as f:
        json.dump(param_sets, f, indent=2)
    
    print(f"Generated {len(param_sets)} parameter combinations")
    """
}

/*
 * Train model with specific parameter set
 */
process trainModel {
    tag "exp_${params_set.experiment_id}"
    publishDir "${params.outdir}/${params_set.experiment_id}", mode: 'copy'
    
    input:
    val params_set
    
    output:
    path "results/*"
    path "logs/*"
    
    script:
    """
    mkdir -p results logs
    
    # Set up environment
    source /Data_large/marine/PythonProjects/SAR/sarpyx/.venv/bin/activate
    cd ${projectDir}
    
    # Run training with parameters
    python main_sweep.py \\
        --experiment_id ${params_set.experiment_id} \\
        --learning_rate ${params_set.learning_rate} \\
        --batch_size ${params_set.batch_size} \\
        --num_layers ${params_set.num_layers} \\
        --hidden_state_size ${params_set.hidden_state_size} \\
        --act_fun ${params_set.act_fun} \\
        --ssim ${params_set.ssim} \\
        --weight_decay ${params_set.weight_decay} \\
        --epochs ${params.epochs} \\
        --valid_batch_size ${params.val_batch_size} \\
        --gpu_no ${params.gpu_no} \\
        --train_dir ${params.train_dir} \\
        --val_dir ${params.val_dir} \\
        --outdir \$PWD/results \\
        --use_wandb true \\
        --wandb_project ssm4sar_sweep \\
        2>&1 | tee logs/training.log
    """
}

/*
 * Aggregate results from all experiments
 */
process aggregateResults {
    publishDir params.outdir, mode: 'copy'
    
    input:
    path 'exp_*/results/*'
    
    output:
    path 'sweep_summary.csv'
    path 'best_params.json'
    
    script:
    """
    #!/usr/bin/env python3
    import json
    import pandas as pd
    import glob
    import os
    
    # Collect all results
    results = []
    
    for exp_dir in glob.glob('exp_*'):
        if os.path.isdir(exp_dir):
            # Try to load metrics
            metrics_file = os.path.join(exp_dir, 'results', 'final_metrics.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    metrics['experiment_id'] = exp_dir
                    results.append(metrics)
    
    if results:
        # Create summary DataFrame
        df = pd.DataFrame(results)
        df.to_csv('sweep_summary.csv', index=False)
        
        # Find best parameters (lowest validation loss)
        if 'val_loss' in df.columns:
            best_idx = df['val_loss'].idxmin()
            best_params = df.iloc[best_idx].to_dict()
            
            with open('best_params.json', 'w') as f:
                json.dump(best_params, f, indent=2)
        
        print(f"Processed {len(results)} experiments")
        print(f"Best validation loss: {df['val_loss'].min():.4f}")
    else:
        print("No results found")
        # Create empty files
        pd.DataFrame().to_csv('sweep_summary.csv', index=False)
        with open('best_params.json', 'w') as f:
            json.dump({}, f)
    """
}

/*
 * Main workflow
 */
workflow {
    // Generate parameter combinations
    param_file = generateParams()
    
    // Parse parameter combinations
    param_sets = param_file
        | splitText() { it.trim() }
        | map { line -> 
            if (line.startsWith('[') || line.startsWith('{')) {
                return new groovy.json.JsonSlurper().parseText(line)
            }
            return null
        }
        | filter { it != null }
        | flatten()
    
    // Train models in parallel
    train_results = trainModel(param_sets)
    
    // Aggregate results
    aggregateResults(train_results.collect())
}

workflow.onComplete {
    log.info """
    Workflow completed!
    Results directory: ${params.outdir}
    """
}
