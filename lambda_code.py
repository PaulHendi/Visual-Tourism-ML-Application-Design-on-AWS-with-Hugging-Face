import json
import boto3
import os
import io


PROMPT_EXAMPLE = "User:{example_prompt}![]({example_image_url})<end_of_utterance>\nAssistant: {example_answer}<end_of_utterance>\n"
PROMPT_TEMPLATE="User:{prompt}![]({image})<end_of_utterance>\nAssistant:"


# Configure the bucket name and the endpoint name 
# with environement variables set in the lambda
BUCKET_NAME = os.environ['BUCKET_NAME']           
URL = f"https://{BUCKET_NAME}.s3.amazonaws.com/"
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']


# Those parameters are fixed for the demo and copied from 
# Philipp's notebook (https://www.philschmid.de/sagemaker-idefics)
PARAMETERS = {
    "do_sample": True,
    "top_p": 0.2,
    "temperature": 0.4,
    "top_k": 50,
    "max_new_tokens": 512,
    "stop": ["User:","<end_of_utterance>"]
}

# This is an example of the prompt that will be sent to the LLM
# One-shot learning is used here, so the prompt is a concatenation of the example prompt and the user prompt
example_prompt = "You are a travel agent. Describe in 1 sentence what is in the image. Add another sentence for an entertaining story related to this location. Add a last sentence saying what is the best place to eat near this location"
example_image_url = f"https://{BUCKET_NAME}.s3.eu-west-3.amazonaws.com/example/trevi.jpeg"
example_answer = "The Trevi Fountain is a popular tourist attraction in Rome, Italy. The act of tossing a coin with one's right hand over the left shoulder is said to ensure a return to Rome, a promise of enduring romance with the city. The best place to eat near this location is the Pinsitaly Trevi restaurant."


prompt_example = PROMPT_EXAMPLE.format(
    example_prompt=example_prompt,
    example_image_url=example_image_url,
    example_answer=example_answer,
)

# In this lambda, we will use : 
# - Sagemaker runtime to invoke the LLM
# - S3 to upload the answer from the LLM
sagemaker_runtime_client = boto3.client('runtime.sagemaker')
s3_client = boto3.client("s3")


def lambda_handler(event, context):
    
    # Retrieve the key of the object that was put into the bucket
    key_image = event['Records'][0]['s3']['object']['key']   
    
    
    # Check if it's an image with the correct format
    file_format = key_image.split(".")[1]
    assert (file_format=="jpeg") or  (file_format=="png") or (file_format=="jpg"),  \
           "File format not covered by the application"
           
           
    # The bucket is public for the demo : we can retrieve the image with an URL       
    image_url = URL + key_image


    # Create the prompt along with an example
    parsed_prompt = PROMPT_TEMPLATE.format(image=image_url,prompt=example_prompt)
    whole_prompt = prompt_example+parsed_prompt
    
    # Create the payload for the invoke request
    payload = json.dumps({"inputs":whole_prompt,"parameters":PARAMETERS})
    
    # Invoke the endpoint
    response = sagemaker_runtime_client.invoke_endpoint(
                                       EndpointName=ENDPOINT_NAME,
                                       ContentType='application/json',
                                       Body=payload)
    
    
    # Decode the response and filter out the answer from the LLM
    prompt_and_result = json.loads(response['Body'].read().decode())
    result = prompt_and_result[0]["generated_text"][len(whole_prompt):].strip()
    
    
    # Upload the result to the bucket with the key output/result_{image_name}.txt
    result_file_object = io.BytesIO(result.encode('utf-8'))
    image_name = key_image.split(".")[0].split("/")[1]
    s3_client.upload_fileobj(result_file_object, 
                             BUCKET_NAME, 
                             f"output/result_{image_name}.txt")
        
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
