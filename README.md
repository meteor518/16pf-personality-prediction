# 16pf-personality-prediction
The input of model is face images and output is the corresponding 16pf personality prediction result. Use dataset to train a model to predict someone's 16pf personality. This is a `multi-label` `multi-class` project.
## 16pf 性格
* 乐群性(gregarious)：高分者外向、热情、乐群；低分者缄默、孤独、内向。
* 聪慧性(intelligent)：高分者聪明、富有才识；低分者迟钝、学识浅薄。
* 稳定性(stable)：高分者情绪稳定而成熟；低分者情绪激动不稳定。
* 恃强性(dominate)：高分者好强固执、支配攻击；低分者谦虚顺从。
* 兴奋性(lively)：高分者轻松兴奋、逍遥放纵；低分者严肃审慎、沉默寡言。
* 有恒性(justice)：高分者有恒负责、重良心；低分者权宜敷衍、原则性差。
* 敢为性(courage)：高分者冒险敢为，少有顾忌，主动性强；低分者害羞、畏缩、退却。
* 敏感性(sensitive)：高分者细心、敏感、好感情用事；低分者粗心、理智、着重实际。
* 怀疑性(suspicious)：高分者怀疑、刚愎、固执己见；低分者真诚、合作、宽容、信赖随和。
* 幻想性(pragmatic)：高分者富于想像、狂放不羁；低分者现实、脚踏实地、合乎成规。
* 世故性(sophisticated)：高分者精明、圆滑、世故、人情练达、善于处世；低分者坦诚、直率、天真。
* 忧虑性(safely)：高分者忧虑抑郁、沮丧悲观、自责、缺乏自信；低分者安详沉着、有自信心。
* 激进性(change)：高分者自由开放、批评激进；低分者保守、循规蹈矩、尊重传统。
* 独立性(independent)：高分者自主、当机立断；低分者依赖、随群附众。
* 自律性(selfcontrol)：高分者知己知彼、自律谨严；低分者不能自制、不守纪律、自我矛盾、松懈、随心所欲。
* 紧张性(gentle)：高分者紧张、有挫折感、常缺乏耐心、心神不定，时常感到疲乏；低分者心平气和、镇静自若、知足常乐。

## 数据格式
标签文件构造格式如`label.csv`中所示，第一列为图片名，剩下的16列为对应的16中性格标签。每个人进行16pf测试后，原本每个指标是0-100的评分，我们将其进行10分制处理作为标签，即，0-10标签为0，10-20标签为1....

## Run
* generate_data.py
```shell
python generate_data.py -i ./images/ -l ./label.csv -o ../train/
```
* train.py
```shell
python train.py -t ./train/images.npy -v ./val/images.npy -tl ./train/label.csv -vl ./val/label.csv -o ../output/ 
```
