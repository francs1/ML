#加载自定义模块
import trainingModel as tm

import dataLoad as dl

def main(argv=None):
	#只执行第一次，将图片的每个像素由整型数据 /255 得到0~1之间的浮点型数据
	#dl.preprecessData()

	mnist = dl.loadData()
	tm.train(mnist)



if __name__ == '__main__':
	main()
	