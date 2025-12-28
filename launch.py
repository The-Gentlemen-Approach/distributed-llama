import os
import sys
import time
import socket
import multiprocessing
from urllib.request import urlopen

def parts(length):
    result = []
    for i in range(length):
        a = chr(97 + (i // 26))
        b = chr(97 + (i % 26))
        result.append(a + b)
    return result

# [['model-url-0', 'model-url-1', ...], 'tokenizer-url', 'weights-float-type', 'buffer-float-type', 'model-type']
MODELS = {
    'llama3_1_8b_instruct_q40': [
        ['https://huggingface.co/b4rtaz/Llama-3_1-8B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_model_llama3.1_instruct_q40.m?download=true'],
        'https://huggingface.co/b4rtaz/Llama-3_1-8B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_tokenizer_llama_3_1.t?download=true',
        'q40', 'q80', 'chat', '--max-seq-len 4096'
    ],
    'llama3_1_405b_instruct_q40': [
        list(map(lambda suffix : f'https://huggingface.co/b4rtaz/Llama-3_1-405B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_model_llama31_405b_q40_{suffix}?download=true', parts(56))),
        'https://huggingface.co/b4rtaz/Llama-3_1-405B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_tokenizer_llama_3_1.t?download=true',
        'q40', 'q80', 'chat', '--max-seq-len 4096'
    ],
    'llama3_2_1b_instruct_q40': [
        ['https://huggingface.co/b4rtaz/Llama-3_2-1B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_model_llama3.2-1b-instruct_q40.m?download=true'],
        'https://huggingface.co/b4rtaz/Llama-3_2-1B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_tokenizer_llama3_2.t?download=true',
        'q40', 'q80', 'chat', '--max-seq-len 4096'
    ],
    'llama3_2_3b_instruct_q40': [
        ['https://huggingface.co/b4rtaz/Llama-3_2-3B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_model_llama3.2-3b-instruct_q40.m?download=true'],
        'https://huggingface.co/b4rtaz/Llama-3_2-3B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_tokenizer_llama3_2.t?download=true',
        'q40', 'q80', 'chat', '--max-seq-len 4096'
    ],
    'llama3_3_70b_instruct_q40': [
        list(map(lambda suffix : f'https://huggingface.co/b4rtaz/Llama-3_3-70B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_model_llama-3.3-70b_q40{suffix}?download=true', parts(11))),
        'https://huggingface.co/b4rtaz/Llama-3_3-70B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_tokenizer_llama-3.3-70b.t?download=true',
        'q40', 'q80', 'chat', '--max-seq-len 4096'
    ],
    'deepseek_r1_distill_llama_8b_q40': [
        ['https://huggingface.co/b4rtaz/DeepSeek-R1-Distill-Llama-8B-Distributed-Llama/resolve/main/dllama_model_deepseek-r1-distill-llama-8b_q40.m?download=true'],
        'https://huggingface.co/b4rtaz/DeepSeek-R1-Distill-Llama-8B-Distributed-Llama/resolve/main/dllama_tokenizer_deepseek-r1-distill-llama-8b.t?download=true',
        'q40', 'q80', 'chat', '--max-seq-len 4096'
    ],
    'qwen3_0.6b_q40': [
        ['https://huggingface.co/b4rtaz/Qwen3-0.6B-Q40-Distributed-Llama/resolve/main/dllama_model_qwen3_0.6b_q40.m?download=true'],
        'https://huggingface.co/b4rtaz/Qwen3-0.6B-Q40-Distributed-Llama/resolve/main/dllama_tokenizer_qwen3_0.6b.t?download=true',
        'q40', 'q80', 'chat', '--max-seq-len 4096'
    ],
    'qwen3_1.7b_q40': [
        ['https://huggingface.co/b4rtaz/Qwen3-1.7B-Q40-Distributed-Llama/resolve/main/dllama_model_qwen3_1.7b_q40.m?download=true'],
        'https://huggingface.co/b4rtaz/Qwen3-1.7B-Q40-Distributed-Llama/resolve/main/dllama_tokenizer_qwen3_1.7b.t?download=true',
        'q40', 'q80', 'chat', '--max-seq-len 4096'
    ],
    'qwen3_8b_q40': [
        ['https://huggingface.co/b4rtaz/Qwen3-8B-Q40-Distributed-Llama/resolve/main/dllama_model_qwen3_8b_q40.m?download=true'],
        'https://huggingface.co/b4rtaz/Qwen3-8B-Q40-Distributed-Llama/resolve/main/dllama_tokenizer_qwen3_8b.t?download=true',
        'q40', 'q80', 'chat', '--max-seq-len 4096'
    ],
    'qwen3_14b_q40': [
        list(map(lambda suffix : f'https://huggingface.co/b4rtaz/Qwen3-14B-Q40-Distributed-Llama/resolve/main/dllama_model_qwen3_14b_q40_{suffix}?download=true', parts(2))),
        'https://huggingface.co/b4rtaz/Qwen3-14B-Q40-Distributed-Llama/resolve/main/dllama_tokenizer_qwen3_14b.t?download=true',
        'q40', 'q80', 'chat', '--max-seq-len 4096'
    ],
    'qwen3_30b_a3b_q40': [
        list(map(lambda suffix : f'https://huggingface.co/b4rtaz/Qwen3-30B-A3B-Q40-Distributed-Llama/resolve/main/dllama_model_qwen3_30b_a3b_{suffix}?download=true', parts(5))),
        'https://huggingface.co/b4rtaz/Qwen3-30B-A3B-Q40-Distributed-Llama/resolve/main/dllama_tokenizer_qwen3_30b_a3b.t?download=true',
        'q40', 'q80', 'chat', '--max-seq-len 4096'
    ],
}

def confirm(message: str):
    alwaysYes = sys.argv.count('-y') > 0
    if alwaysYes:
        return True
    result = input(f'❓ {message} ("Y" if yes): ').upper()
    return result == 'Y' or result == 'YES'

def downloadFile(urls, path: str):
    if os.path.isfile(path):
        fileName = os.path.basename(path)
        if not confirm(f'{fileName} already exists, do you want to download again?'):
            return

    socket.setdefaulttimeout(30)
    lastSizeMb = 0
    with open(path, 'wb') as file:
        for url in urls:
            startPosition = file.tell()
            success = False
            for attempt in range(8):
                print(f'📄 {url} (attempt: {attempt})')
                try:
                    with urlopen(url) as response:
                        while True:
                            chunk = response.read(4096)
                            if not chunk:
                                break
                            file.write(chunk)
                            sizeMb = file.tell() // (1024 * 1024)
                            if sizeMb != lastSizeMb:
                                sys.stdout.write("\rDownloaded %i MB" % sizeMb)
                                lastSizeMb = sizeMb
                    sys.stdout.write('\n')
                    success = True
                    break
                except Exception as e:
                    print(f'\n❌ Error downloading {url}: {e}')
                file.seek(startPosition)
                file.truncate()
                time.sleep(1 * attempt)
            if not success:
                raise Exception(f'Failed to download {url}')
    sys.stdout.write(' ✅\n')

def download(modelName: str, model: list):
    dirPath = os.path.join('models', modelName)
    print(f'📀 Downloading {modelName} to {dirPath}...')
    os.makedirs(dirPath, exist_ok=True)
    modelUrls = model[0]
    tokenizerUrl = model[1]
    modelPath = os.path.join(dirPath, f'dllama_model_{modelName}.m')
    tokenizerPath = os.path.join(dirPath, f'dllama_tokenizer_{modelName}.t')
    downloadFile(modelUrls, modelPath)
    downloadFile([tokenizerUrl], tokenizerPath)
    print('📀 All files are downloaded')
    return (modelPath, tokenizerPath)

def getMode():
    """Get execution mode from command line arguments"""
    for arg in sys.argv:
        if arg.startswith('--mode='):
            return arg.split('=')[1]
    return 'simple'  # Default to simple mode

def getWorkerCount():
    """Get worker count for hpipe mode"""
    for arg in sys.argv:
        if arg.startswith('--workers='):
            return int(arg.split('=')[1])
    return 2  # Default to 2 workers

def writeRunFile(modelName: str, mode: str, modelPath: str, tokenizerPath: str, bufferType: str, extraArgs: str, nThreads: int, workerCount: int):
    """Write execution script based on mode"""
    filePath = f'run_{modelName}_{mode}.sh'

    with open(filePath, 'w') as file:
        file.write('#!/bin/bash\n')
        file.write(f'# {modelName} - {mode.upper()} mode\n')
        file.write('\n')

        if mode == 'simple':
            # Simple single-node execution
            file.write(f'./simple-dllama \\\n')
            file.write(f'  --model {modelPath} \\\n')
            file.write(f'  --tokenizer {tokenizerPath} \\\n')
            file.write(f'  --buffer-float-type {bufferType} \\\n')
            file.write(f'  --nthreads {nThreads} \\\n')
            file.write(f'  --steps 100 \\\n')
            file.write(f'  --prompt "Once upon a time"')
            if extraArgs:
                file.write(f' \\\n  {extraArgs}')
            file.write('\n')

        elif mode == 'hpipe':
            # H-Pipe pipeline parallelism
            file.write('# Start workers in background\n')
            for i in range(workerCount):
                port = 9999 + i
                file.write(f'./hpipe-worker --port {port} --nthreads {nThreads} > worker{i}.log 2>&1 &\n')

            file.write('\n# Wait for workers to start\n')
            file.write('sleep 3\n')
            file.write('\n# Run root coordinator\n')

            workers_str = ' '.join([f'127.0.0.1:{9999+i}' for i in range(workerCount)])
            file.write(f'./hpipe-root \\\n')
            file.write(f'  --model {modelPath} \\\n')
            file.write(f'  --tokenizer {tokenizerPath} \\\n')
            file.write(f'  --workers {workers_str} \\\n')
            file.write(f'  --nthreads {nThreads} \\\n')
            file.write(f'  --steps 50 \\\n')
            file.write(f'  --chunk-size 64 \\\n')
            file.write(f'  --prompt "Once upon a time"')
            if extraArgs:
                file.write(f' \\\n  {extraArgs}')
            file.write('\n')
            file.write('\n# Cleanup\n')
            file.write('pkill -f hpipe-worker\n')

    # Make executable
    os.chmod(filePath, 0o755)
    return filePath

def printUsage():
    print('Usage: python launch.py <model> [options]')
    print()
    print('Options:')
    print('  <model>         The name of the model to download')
    print('  --mode=MODE     Execution mode: simple (default), hpipe')
    print('  --workers=N     Number of workers for hpipe mode (default: 2)')
    print('  -skip-run       Do not run the model after download')
    print('  -skip-script    Do not create a script to run the model')
    print('  -y              Skip confirmation prompts')
    print()
    print('Execution modes:')
    print('  simple   - Single-node execution (fastest to start, good for testing)')
    print('  hpipe    - Pipeline parallelism (token-level, any number of workers)')
    print()
    print('Available models:')
    for model in MODELS:
        print(f'  {model}')
    print()
    print('Examples:')
    print('  python launch.py llama3_2_1b_instruct_q40')
    print('  python launch.py llama3_2_1b_instruct_q40 --mode=hpipe --workers=3')

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        printUsage()
        exit(1)

    os.chdir(os.path.dirname(__file__))

    modelName = sys.argv[1].replace('-', '_')
    if modelName not in MODELS:
        print(f'Model is not supported: {modelName}')
        exit(1)

    model = MODELS[modelName]
    (modelPath, tokenizerPath) = download(modelName, model)

    mode = getMode()
    workerCount = getWorkerCount()
    nThreads = multiprocessing.cpu_count()

    extraArgs = model[5] if len(model) > 5 else ''

    # Build and display command
    print()
    print(f'🎯 Mode: {mode.upper()}')
    if mode == 'hpipe':
        print(f'👥 Workers: {workerCount}')
    print(f'🧵 Threads: {nThreads}')
    print()

    skipRun = sys.argv.count('-skip-run') > 0
    skipScript = sys.argv.count('-skip-script') > 0

    if (not skipScript):
        runFilePath = writeRunFile(modelName, mode, modelPath, tokenizerPath,
                                   model[3], extraArgs, nThreads, workerCount)
        print(f'📝 Created {runFilePath} script for easy execution')
        print(f'   Run with: ./{runFilePath}')
        print()

    if (not skipRun):
        if (confirm(f'Do you want to run in {mode} mode now?')):
            # Build the appropriate binary
            if mode == 'simple':
                if not os.path.isfile('simple-dllama'):
                    print('🔨 Building simple-dllama...')
                    os.system('make simple-dllama')
                print('🚀 Running simple-dllama...')
                command = f'./simple-dllama --model {modelPath} --tokenizer {tokenizerPath} --buffer-float-type {model[3]} --nthreads {nThreads} --steps 100 --prompt "Once upon a time"'
                if extraArgs:
                    command += f' {extraArgs}'
                os.system(command)

            elif mode == 'hpipe':
                if not os.path.isfile('hpipe-root') or not os.path.isfile('hpipe-worker'):
                    print('🔨 Building hpipe binaries...')
                    os.system('make hpipe-root hpipe-worker')
                print(f'🚀 Running H-Pipe with {workerCount} workers...')
                runFilePath = f'run_{modelName}_hpipe.sh'
                os.system(f'./{runFilePath}')

            else:
                print(f'❌ Unknown mode: {mode}')
                print('   Supported modes: simple, hpipe')
                exit(1)
