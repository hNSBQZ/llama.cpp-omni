好的，现在我这个分支是用来把llm到tts中间的部分从cpu变为算子图具体行动位于/cache/hanqingzhe/Video-MME/llama.cpp-omni/docs/tts_llm_to_tts_flow.md 现在需要你完成后续验证工作
1.尝试在/cache/hanqingzhe/Video-MME/llama.cpp-omni/tools/omni/omni-impl.h增加一个dumptensor函数，可以将[dim，len，batch，1]的tensor用6位小数点的方式打印到文件上，格式为一行有dim个数字，中间用空格分开，每行用换行符隔开，有len*batch行，这样方便人眼和机器同时debug
2.写一个python脚本，环境依赖很简单，像cppeval什么的虚拟环境就行，可以输入两个文件名，文件里是1中保存的张量，对比两个张量，先形状，形状不对直接报错结束，然后是对比精度差异，每个位置上的差取绝对值，看整个张量的平均值，最大值，最小值，标准差并打印出来
3.交给你这个cli自动测评吧，跑一个/cache/hanqingzhe/Video-MME/llama.cpp-omni/tools/omni/test/single_test_omni.cpp（记得修改参数和路径再编译），跑一边保存每次的运行日志，第一次是用常规的cpu算子，第二次用我gpu改写的算子，对比精度差异和时间差异
4.将最终的结果以md记录