from io import BytesIO

async def read_file(file):
    try:
        buffer = BytesIO()

        while content := file.read(1024 * 1024):  # Read file in chunks of 1MB  
            buffer.write(content)
            buffer.seek(0)
        
    except Exception as e:
        raise e
    
    return buffer