"""
Qdrant Vector Database Reset Utility

Options:
1. Delete and recreate the collection (keeps Qdrant running)
2. Stop and remove Qdrant container (full reset)
3. List all collections
"""

import requests
from qdrant_client import QdrantClient
import argparse
import time


QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "research_memory"


def delete_collection():
    """Delete the collection from Qdrant"""
    print(f"🗑️  Deleting collection '{COLLECTION_NAME}'...")
    
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.delete_collection(collection_name=COLLECTION_NAME)
        print("✅ Collection deleted successfully")
        return True
    except Exception as e:
        print(f"❌ Error deleting collection: {e}")
        return False


def list_collections():
    """List all collections in Qdrant"""
    print("📊 Listing all Qdrant collections...\n")
    
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = client.get_collections()
        
        if not collections.collections:
            print("No collections found")
            return
        
        print(f"{'Collection Name':<30} {'Vectors':<15} {'Points':<15}")
        print("-" * 60)
        
        for collection in collections.collections:
            name = collection.name
            vectors_count = getattr(collection, 'points_count', 0)
            print(f"{name:<30} {vectors_count:<15}")
        
        print(f"\nTotal collections: {len(collections.collections)}")
    except Exception as e:
        print(f"❌ Error listing collections: {e}")


def check_qdrant():
    """Check if Qdrant is running"""
    print("🔍 Checking Qdrant status...")
    
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        info = client.get_collections()
        print("✅ Qdrant is running and responding")
        print(f"   Collections: {len(info.collections)}")
        return True
    except Exception as e:
        print(f"❌ Qdrant is not running or not accessible: {e}")
        print("\nStart Qdrant with:")
        print("  docker run -p 6333:6333 qdrant/qdrant")
        return False


def full_reset():
    """Full reset: stop and remove Qdrant container"""
    print("🔄 FULL RESET - Stop and remove Qdrant container")
    print("This will delete ALL data in Qdrant")
    
    response = input("\nAre you sure? Type 'YES' to confirm: ")
    if response != "YES":
        print("❌ Cancelled")
        return
    
    print("\n1️⃣  Stopping Qdrant container...")
    import subprocess
    
    try:
        # Stop container
        subprocess.run(
            ["docker", "stop", "$(docker ps -q --filter ancestor=qdrant/qdrant)"],
            shell=True,
            check=False,
        )
        print("✅ Qdrant stopped")
        
        time.sleep(2)
        
        # Remove container
        print("2️⃣  Removing Qdrant container...")
        subprocess.run(
            ["docker", "rm", "$(docker ps -aq --filter ancestor=qdrant/qdrant)"],
            shell=True,
            check=False,
        )
        print("✅ Container removed")
        
        print("\n3️⃣  Starting fresh Qdrant...")
        subprocess.Popen(
            ["docker", "run", "-p", "6333:6333", "qdrant/qdrant"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("✅ New Qdrant instance starting")
        
        time.sleep(3)
        
        if check_qdrant():
            print("\n✨ Full reset complete - Qdrant is clean and ready!")
        else:
            print("\n⚠️  Qdrant is starting up, may take a few seconds...")
            
    except Exception as e:
        print(f"❌ Error during full reset: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Qdrant Database Reset Utility"
    )
    parser.add_argument(
        "--action",
        choices=["delete-collection", "list", "check", "full-reset"],
        default="check",
        help="Action to perform (default: check)",
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🗄️  QDRANT RESET UTILITY")
    print("="*70 + "\n")
    
    if args.action == "check":
        check_qdrant()
        print("\nAvailable collections:")
        list_collections()
    
    elif args.action == "list":
        list_collections()
    
    elif args.action == "delete-collection":
        response = input(
            f"Delete collection '{COLLECTION_NAME}'? Type 'YES' to confirm: "
        )
        if response == "YES":
            if delete_collection():
                print("✅ Ready to re-upload PDFs and start fresh")
        else:
            print("❌ Cancelled")
    
    elif args.action == "full-reset":
        full_reset()
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
