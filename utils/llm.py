import time
import threading
import os

from dotenv import load_dotenv
load_dotenv()

from fireworks.client import Fireworks

class BaseLLM:
  """Base LLM class."""
  def __init__(self) -> None:
    self.counter_llm_calls = 0
    self._lock = threading.Lock()
    pass

  def increment_call_counter(self):
    with self._lock:
      self.counter_llm_calls += 1

  def get_call_counter(self):
    with self._lock:
      return self.counter_llm_calls
    
  def reset_call_counter(self):
    with self._lock:
      self.counter_llm_calls = 0

  def predict(self, **kwargs) -> str:
    pass


class LLM(BaseLLM):
  """OpenAI LLM class. This serves as a wrapper for the OpenAI API."""

  model_name:str
  temperature:int
  request_timeout:int
  multithreaded:bool

  def __init__(self, model_name, temperature, request_timeout, multithreaded=True, **kwargs) -> None:
    super().__init__()
    self.model_name = model_name
    self.temperature = temperature
    self.request_timeout = request_timeout
    self.multithreaded = multithreaded
  
  def predict(self, user_prompt, logger, **kwargs) -> str:
    """Makes a call to the API. Double the timeout time if the request times out."""
    self.increment_call_counter()

    wait = 20
    completion = None
    while wait < self.request_timeout:
      try:
        if self.model_name == 'fireworks-llama-v3-70b-instruct':
          client = Fireworks(api_key=os.getenv('FIREWORKS_API_KEY'))
          response = client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3-70b-instruct",
            messages=[
              {"role": "user",
               "content": user_prompt},
            ],
          ).choices[0].message.content  

          if logger is not None: logger.info(f'Fireworks API Call. Finish Model: {self.model_name}. User Prompt: {user_prompt}, Output: {response}')

        elif self.model_name == 'fireworks-llama-v3-8b-instruct':
          client = Fireworks(api_key=os.getenv('FIREWORKS_API_KEY'))
          response = client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3-8b-instruct",
            messages=[
              {"role": "user",
               "content": user_prompt},
            ],
          ).choices[0].message.content  

          formatted_response = response.replace('\\n', '. ')
          if logger is not None: logger.info(f'Fireworks API Call. Finish Model: {self.model_name}. User Prompt: {user_prompt}, Output: {formatted_response}')

        if self.model_name == 'fireworks-llama-v3p1-8b-instruct':
          client = Fireworks(api_key=os.getenv('FIREWORKS_API_KEY'))
          response = client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p1-8b-instruct",
            messages=[
              {"role": "user",
               "content": user_prompt},
            ],
          ).choices[0].message.content  

          formatted_response = response.replace('\\n', '. ')
          if logger is not None: logger.info(f'Fireworks API Call. Finish Model: {self.model_name}. User Prompt: {user_prompt}, Output: {formatted_response}')          

        else:
          if logger is not None: logger.critical('The specified model does not exist.')
          raise Exception('The specified model does not exist.') 
        
        return response

      except Exception as e:
          print(e)
          if logger is not None: logger.error(f'class oaiLLM/predict(), Error, {e}')
          if logger is not None: logger.info(f'Waiting for {str(wait)} seconds...')
          time.sleep(wait)
          wait *= 2
          continue
    if logger is not None: logger.error('The request timed out.')
    raise Exception('The request timed out.')