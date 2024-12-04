# cli.py
from app import PharmaKnowledgeAssistant

def main():
    assistant = PharmaKnowledgeAssistant()
    
    print("Welcome to Pharma Knowledge Assistant!")
    print("Type 'quit' to exit")
    
    while True:
        query = input("\nYour query: ")
        
        if query.lower() == 'quit':
            break
            
        response = assistant.process_query(query)
        print("\nAssistant:", response)

if __name__ == "__main__":
    main()