# NYC Short-Term Rental Price Prediction Pipeline

An automated ML pipeline for predicting short-term rental prices based on property characteristics. The pipeline processes weekly data updates and retrains the model automatically.

## System Requirements

- **Operating Systems**:
  - Ubuntu 22.04/24.04 (native or WSL)
  - Recent macOS versions
- **Python**: Version 3.10
- **Package Manager**: Conda

## Quick Start

1. Clone the repository:
```bash
git clone [your repository URL]
cd [repository name]
```

2. Set up the environment:
```bash
conda env create -f environment.yml
conda activate nyc_airbnb_dev
```

3. Configure Weights & Biases:
- Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize)
- Login:
```bash
wandb login [your API key]
```

4. Run the pipeline:
```bash
mlflow run .
```

## Pipeline Configuration

Modify `config.yaml` to adjust pipeline parameters. Key configurations:
- Data sampling rate
- Price thresholds
- Model hyperparameters
- Feature engineering settings

## Running Specific Pipeline Steps

Execute individual steps:
```bash
mlflow run . -P steps=download
```

Run multiple steps:
```bash
mlflow run . -P steps=download,basic_cleaning
```

Override config parameters:
```bash
mlflow run . -P hydra_options="modeling.random_forest.n_estimators=10 etl.min_price=50"
```

## Troubleshooting

### Environment Issues
Clean up corrupted MLflow environments:
```bash
for e in $(conda info --envs | grep mlflow | cut -f1 -d" "); do conda uninstall --name $e --all -y;done
```

### Common Errors
- Ensure Python 3.10 is the default version
- Verify MLflow and Weights & Biases versions match
- Check conda is properly installed and initialized

## License

This project is licensed under the MIT License - see the [MIT License](https://opensource.org/licenses/MIT) for details.

## Project Links

- Weights & Biases Dashboard: https://wandb.ai/stephenmink65-western-governors-university/nyc_airbnb
- GitHub Repository: https://github.com/stephenmink/Project-Build-an-ML-Pipeline-Starter