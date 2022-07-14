```python
if __name__ == '__main__':
    code_dir = os.path.dirname(__file__)
    work_dir = os.path.join(code_dir, '../../')
    sys.path.append(work_dir)
    print('>>>>>code_dir:{}, work_dir:{}'.format(code_dir, work_dir))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_url", type=str, default="./dataset")
    parser.add_argument("--train_url", type=str, default="./output")
    current_path = os.getcwd()
    print('>>>>>>>>>>>>>>current_path:{}<<<<<<<<'.format(current_path))
    config = parser.parse_args()
    data_dir = "/cache/dataset"
    os.makedirs(data_dir)
    model_dir = "/cache/result"
    os.makedirs(model_dir)
    mox.file.copy_parallel(config.data_url, data_dir)
    app.run(main)
    mox.file.copy_parallel(model_dir, config.train_url)

```