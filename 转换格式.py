def convert_to_demo_format(input_filename="submission1.txt", output_filename="answer14.txt"):
    """
    将提交结果文件转换为 demo 格式。
    即去除每行开头的 ID，只保留后面的四元组信息。
    """
    with open(input_filename, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for line in lines:
            # 找到第一个制表符的位置
            tab_index = line.find('\t')
            if tab_index != -1:
                # 提取制表符后面的内容，并去除首尾空白（包括换行符）
                output_content = line[tab_index + 1:].strip()
                outfile.write(output_content + '\n')
            else:
                # 如果没有制表符，说明格式可能不对，但为了不丢失数据，直接写入整行
                outfile.write(line.strip() + '\n')

    print(f"转换完成！结果已保存到 {output_filename}")

# 运行转换函数
convert_to_demo_format()