from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def handle_conversation():
    context = ""
    message = ""

    template = f"""
    You are a product description generator. Based on the following message, generate a detailed and creative product description:
    
    Here is the context - {context}
    
    Input - {message}
    
    Provide information about the key features, benefits, target audience, and unique selling points.

    For example:
    Input - 
    Create a description for product code WFO with following attributes - 
        Name - 'Optimus - 20 oz. Double Wall Stainless Tumbler with Ceramic Interior'
        Material(Inner & Outer) - Recycled 304 Stainless Steel, Lid Clear AS
        Weight - 283.20 g
        Dimensions - 19.2 cm height with lid x 8.6 cm diameter
        Capacity - 20 oz
        Lid - Clear
        Color - Black/Blue/Green

    Response - 
    Discover the Optimus, a 20 oz. double wall stainless steel tumbler designed to elevate your beverage experience. Featuring double wall vacuum insulation and a ceramic coating inside, this tumbler keeps your hot drinks warm for up to 6 hours and your cold drinks chilled for up to 12 hours.
    Crafted from recycled 304 stainless steel, the Optimus combines sustainability with functionality. The outer body features a matte coating with an upper part in a sleek metal silver finish, offering a stylish and modern look. The clear AS lid securely presses on to prevent spills while allowing easy sipping.
    Weighing 283.2 grams, the Optimus is not suitable for dishwashers or microwaves. It comes with an instruction manual inside, ensuring you get the most out of its features.
    Ideal for both hot and cold beverages, the Optimus is perfect for everyday use, whether at home, in the office, or on the go.
    """

    model = OllamaLLM(model="llama3.1:latest")
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    print("Welcome to Product Description Generator. Type 'exit' anytime to quit.")
    while True:
        user_input = input("Enter the product details: ")
        if user_input.lower() == "exit":
            break
        result = chain.invoke({"context": context, "message": user_input})
        print("Product Description Generator: ", result)
        context += f"\nUser: {user_input}\nAI: {result}"

if __name__ == "__main__":
    handle_conversation()
