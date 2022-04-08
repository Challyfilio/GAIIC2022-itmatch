下载数据，放到data文件夹下：
```
data
├── attr_to_attrvals.json
├── test.txt
├── train_coarse.txt
└── train_fine.txt
```
分别为属性字典文件、测试文件、粗标数据、细标数据。

---

划分数据：

```
split -l 49000 -d data/train_fine.txt data/train_fine.txt.
```
`data/train_fine.txt.01`用于模型验证。