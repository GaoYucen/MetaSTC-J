from optimized_runner import run_experiment

if __name__ == "__main__":
    run_experiment({"family": "lstm", "dataset": "beijing", "clusters": 5, "batch_size": 8192, "output_dir": "param/4090_tuned/lstm/beijing"})
