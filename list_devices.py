import sounddevice as sd

print("\n🔊 Available Audio Devices:\n")
print(sd.query_devices())

print("\n🎙️ Default input/output device index:")
print(sd.default.device)
