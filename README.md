Terminal1(brocker):
'''
mosquitto -v
'''

Terminal2(subscriber):
'''
uv run python processor/subscriber.py
'''

Terminal3(publisher)
'''
uv run python run.py
'''