import os.path

import torch
import xlwt
from torch import nn
from torch.utils.data import DataLoader

from dataset import ValidateRadarData, device
from net import validate

mdl_path = 'model_backup.pt'
split = False
apply_window = False
refusion = 0.6

postprocess = nn.Softmax(dim=1)


def main():
    while True:
        file = input("输入测试文件夹(validData文件夹)路径: ")
        if os.path.exists(file):
            break
        else:
            print("该目录不存在!\n")
    print("加载数据集...")
    dataset = ValidateRadarData(file, split=split, apply_window=apply_window)
    dataloader = DataLoader(dataset, batch_size=2)
    print("加载模型...")
    mdl = torch.load(mdl_path, map_location=torch.device(device=device))
    print("开始验证...")
    results = validate(dataloader, mdl, postprocess, refusion=refusion)
    print("正在生成表格...")
    book = xlwt.Workbook(encoding='utf-8')
    sheet = book.add_sheet('结果表格')
    sheet.write(0, 0, '序号')
    sheet.write(0, 1, '测试数据块名称')
    sheet.write(0, 2, '识别结果')
    sheet.write(0, 3, '单个数据块识别概率')
    for i in range(len(results)):
        sheet.write(i + 1, 0, i + 1)
        sheet.write(i + 1, 1, str(results[i][0]))
        sheet.write(i + 1, 2, int(results[i][1]))
        sheet.write(i + 1, 3, f"{results[i][2]:>6}")

    book.save("./结果.xls")
    print("完成! 表格保存在当前目录下的 结果.xls")


if __name__ == '__main__':
    main()
