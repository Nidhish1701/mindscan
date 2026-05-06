"""
Mental Health Early Warning System — Unified CLI Entry Point.

Commands:
    python run.py train          — Train DistilBERT transformer model
    python run.py train-cnn      — Train CNN-LSTM-Attention model
    python run.py train-xgb      — Train XGBoost baseline
    python run.py serve          — Start FastAPI server
    python run.py predict        — Quick prediction from command line
    python run.py check          — Validate environment & imports

Usage:
    python run.py train --sample 20000 --epochs 3
    python run.py serve --port 8000
    python run.py predict --text "I feel hopeless"
"""

import os
import sys
import argparse


def cmd_train(args):
    """Train the DistilBERT transformer model."""
    from src.training.train_transformer import train_transformer
    train_transformer(
        data_path=args.data,
        model_name=args.model_name,
        output_dir=args.output,
        sample_size=args.sample,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        patience=args.patience,
        fp16=not args.no_fp16,
    )


def cmd_train_cnn(args):
    """Train the CNN-LSTM-Attention model."""
    from src.training.train_cnn_lstm import train_cnn_lstm
    train_cnn_lstm(
        data_path=args.data,
        output_dir=args.output,
        sample_size=args.sample,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
    )


def cmd_train_xgb(args):
    """Train the XGBoost baseline."""
    from src.training.train_xgboost import train_xgboost
    train_xgboost(
        data_path=args.data,
        output_path=args.output,
        sample_size=args.sample,
    )


def cmd_serve(args):
    """Start the FastAPI server."""
    import uvicorn
    print("=" * 60)
    print("  Mental Health Early Warning System — API Server")
    print("=" * 60)
    print(f"  URL: http://localhost:{args.port}")
    print(f"  Docs: http://localhost:{args.port}/docs")
    print(f"  Dashboard: http://localhost:{args.port}/")
    print("=" * 60)
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def cmd_predict(args):
    """Quick prediction from command line."""
    from src.inference.predict import load_model, predict
    from src.inference.risk_scorer import calculate_risk_score

    model_dir = args.model_dir
    model, tokenizer, label_encoder = load_model(model_dir)
    results = predict([args.text], model, tokenizer, label_encoder)
    r = results[0]

    risk = calculate_risk_score(
        text=args.text,
        prediction=r['prediction'],
        confidence=r['confidence'],
        probabilities=r['probabilities'],
    )

    print(f"\n{'='*56}")
    print(f"  Text:        {args.text[:60]}...")
    print(f"  Prediction:  {r['prediction'].upper()}")
    print(f"  Confidence:  {r['confidence']*100:.1f}%")
    print(f"  Risk Score:  {risk['risk_score']}/100 ({risk['severity']})")
    print(f"  Crisis Flag: {'⚠ YES' if risk['is_crisis'] else '✓ No'}")
    print(f"\n  Probabilities:")
    for cls, prob in sorted(r['probabilities'].items(), key=lambda x: -x[1]):
        bar = '█' * int(prob * 30)
        print(f"    {cls:<20} {prob*100:5.1f}%  {bar}")
    print(f"\n  Recommendations:")
    for rec in risk['recommendations']:
        print(f"    • {rec}")
    print(f"{'='*56}")


def cmd_check(args):
    """Validate environment, imports, and dependencies."""
    print("Checking Python version ...")
    print(f"  Python {sys.version}")
    assert sys.version_info >= (3, 9), "Python 3.9+ required"
    print("  ✓ OK\n")

    checks = [
        ("torch",        "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"),
        ("transformers", "import transformers; print(f'  Transformers {transformers.__version__}')"),
        ("sklearn",      "import sklearn; print(f'  Scikit-learn {sklearn.__version__}')"),
        ("fastapi",      "import fastapi; print(f'  FastAPI {fastapi.__version__}')"),
        ("xgboost",      "import xgboost; print(f'  XGBoost {xgboost.__version__}')"),
        ("pandas",       "import pandas; print(f'  Pandas {pandas.__version__}')"),
        ("numpy",        "import numpy; print(f'  NumPy {numpy.__version__}')"),
    ]

    all_ok = True
    for name, code in checks:
        try:
            exec(code)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} — {e}")
            all_ok = False

    # Check project structure
    print("\nChecking project structure ...")
    required_dirs = ['src', 'src/preprocessing', 'src/training', 'src/inference', 'src/api', 'src/chatbot', 'frontend', 'data']
    for d in required_dirs:
        exists = os.path.isdir(d)
        print(f"  {'✓' if exists else '✗'} {d}/")
        if not exists:
            all_ok = False

    # Check dataset
    print("\nChecking dataset ...")
    data_file = "data/mental_disorders_reddit.csv"
    if os.path.exists(data_file):
        size_mb = os.path.getsize(data_file) / (1024 * 1024)
        print(f"  ✓ {data_file} ({size_mb:.0f} MB)")
    else:
        print(f"  ✗ {data_file} not found")
        all_ok = False

    print(f"\n{'✓ All checks passed!' if all_ok else '✗ Some checks failed.'}")


def main():
    parser = argparse.ArgumentParser(
        description="Mental Health Early Warning System — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # ---- train ----
    p_train = subparsers.add_parser('train', help='Train DistilBERT model')
    p_train.add_argument('--data',       default='data/mental_disorders_reddit.csv')
    p_train.add_argument('--model_name', default='distilbert-base-uncased')
    p_train.add_argument('--output',     default='models/distilbert/best_model')
    p_train.add_argument('--sample',     type=int, default=None, help='Sample N rows')
    p_train.add_argument('--epochs',     type=int, default=4)
    p_train.add_argument('--batch_size', type=int, default=32)
    p_train.add_argument('--lr',         type=float, default=2e-5)
    p_train.add_argument('--max_length', type=int, default=128)
    p_train.add_argument('--patience',   type=int, default=2)
    p_train.add_argument('--no_fp16',    action='store_true')

    # ---- train-cnn ----
    p_cnn = subparsers.add_parser('train-cnn', help='Train CNN-LSTM-Attention model')
    p_cnn.add_argument('--data',       default='data/mental_disorders_reddit.csv')
    p_cnn.add_argument('--output',     default='models/cnn_lstm')
    p_cnn.add_argument('--sample',     type=int, default=None)
    p_cnn.add_argument('--epochs',     type=int, default=10)
    p_cnn.add_argument('--batch_size', type=int, default=64)
    p_cnn.add_argument('--lr',         type=float, default=1e-3)
    p_cnn.add_argument('--patience',   type=int, default=3)

    # ---- train-xgb ----
    p_xgb = subparsers.add_parser('train-xgb', help='Train XGBoost baseline')
    p_xgb.add_argument('--data',    default='data/mental_disorders_reddit.csv')
    p_xgb.add_argument('--output',  default='models/xgb_model.pkl')
    p_xgb.add_argument('--sample',  type=int, default=None)

    # ---- serve ----
    p_serve = subparsers.add_parser('serve', help='Start API server')
    p_serve.add_argument('--host',   default='0.0.0.0')
    p_serve.add_argument('--port',   type=int, default=8000)
    p_serve.add_argument('--reload', action='store_true')

    # ---- predict ----
    p_pred = subparsers.add_parser('predict', help='Quick prediction')
    p_pred.add_argument('--text',      required=True, help='Text to classify')
    p_pred.add_argument('--model_dir', default='models/distilbert/best_model')

    # ---- check ----
    subparsers.add_parser('check', help='Validate environment')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        'train': cmd_train,
        'train-cnn': cmd_train_cnn,
        'train-xgb': cmd_train_xgb,
        'serve': cmd_serve,
        'predict': cmd_predict,
        'check': cmd_check,
    }

    commands[args.command](args)


if __name__ == '__main__':
    main()
