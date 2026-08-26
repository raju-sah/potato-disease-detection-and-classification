#!/usr/bin/env python3
"""
Deploy Potato Leaf Disease Classification Web App to Hugging Face Spaces.
Target User: raju-ai (https://huggingface.co/raju-ai)
Space URL: https://huggingface.co/spaces/raju-ai/potato-leaf-disease-classifier
"""

import os
import sys
import argparse
import subprocess

def check_or_install_huggingface_hub():
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError:
        print("📦 Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "huggingface_hub"])
        import huggingface_hub
        return huggingface_hub

def deploy(token=None, space_name="potato-leaf-disease-classifier", username="raju-ai", telegram_token=None, telegram_chat_id=None):
    hf = check_or_install_huggingface_hub()
    from huggingface_hub import HfApi, create_repo, upload_folder
    
    hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    api = HfApi(token=hf_token)
    repo_id = f"{username}/{space_name}"
    space_url = f"https://huggingface.co/spaces/{repo_id}"
    
    print(f"\n🚀 Deploying to Hugging Face Space: {repo_id}")
    print(f"🔗 Target URL: {space_url}\n")
    
    try:
        # Create Space if it doesn't exist
        print(f"Creating / verifying space '{repo_id}' (SDK: Docker)...")
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            token=hf_token,
            exist_ok=True
        )
        print("✅ Space repository ready.")
        
        # Upload project files
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Uploading files from {repo_dir} to {repo_id}...")
        
        upload_folder(
            folder_path=repo_dir,
            repo_id=repo_id,
            repo_type="space",
            token=hf_token,
            ignore_patterns=[
                ".venv/**",
                ".git/**",
                "__pycache__/**",
                "*.pyc",
                ".DS_Store",
                "kaggle-output/**",
                ".agents/**"
            ]
        )
        
        print(f"\n🎉 Deployment successful!")
        print(f"🌐 Access your live application at: {space_url}")
        
        # Send Telegram notification if credentials provided
        t_token = telegram_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        t_chat = telegram_chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        
        if t_token and t_chat:
            try:
                import urllib.request
                import urllib.parse
                msg = f"🥔 *Potato Leaf Disease Classifier Deployed!*\n\n🚀 Live Hugging Face Space: {space_url}\nStatus: Built and Serving with FastAPI + TFLite!"
                tg_url = f"https://api.telegram.org/bot{t_token}/sendMessage?chat_id={t_chat}&text={urllib.parse.quote(msg)}&parse_mode=Markdown"
                urllib.request.urlopen(tg_url)
                print("📱 Telegram notification sent successfully!")
            except Exception as tg_err:
                print(f"⚠️ Telegram ping failed: {tg_err}")
                
        return space_url
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        print("\n💡 Tip: Provide your Hugging Face Access Token with 'write' permission:")
        print(f"   HF_TOKEN=hf_xxx python deploy_hf.py")
        print(f"   Or run: hf auth login\n")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy to Hugging Face Spaces")
    parser.add_argument("--token", help="Hugging Face API Write Token", default=None)
    parser.add_argument("--space", help="Space name", default="potato-leaf-disease-classifier")
    parser.add_argument("--username", help="HF Username", default="raju-ai")
    parser.add_argument("--telegram-token", help="Telegram Bot Token", default=None)
    parser.add_argument("--telegram-chat-id", help="Telegram Chat ID", default=None)
    
    args = parser.parse_args()
    deploy(
        token=args.token,
        space_name=args.space,
        username=args.username,
        telegram_token=args.telegram_token,
        telegram_chat_id=args.telegram_chat_id
    )
