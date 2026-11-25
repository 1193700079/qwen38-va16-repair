"""Public request policy: no unverified paths may enter the worker."""
import pytest
from concurrency_gate import validate_body

def test_text_greedy_allowed():
    assert validate_body({'messages':[{'role':'user','content':'你好'}],'temperature':0}) is None

def test_random_sampling_rejected_before_engine():
    assert validate_body({'temperature':.7}) is not None

def test_multiple_outputs_rejected_before_engine():
    assert validate_body({'n':2}) is not None

def test_image_input_rejected_before_engine():
    assert validate_body({'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'https://example.org/a.jpg'}}]}]}) is not None

def test_structured_decode_rejected_before_engine():
    assert validate_body({'tools':[{'type':'function'}]}) is not None
