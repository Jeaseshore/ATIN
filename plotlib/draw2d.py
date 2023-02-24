'''
@Project : math2022lab
@File    : draw2d.py
@Author  : Qing Zhang
@Date    : 2022/9/11 15:03
'''

import matplotlib.pyplot as plt
import numpy as np

# 使之能绘制中文
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def draw_line(x, y, title, x_axis_names, y_axis_names, fig_num=1, subgraph_dis=11
              , figsize=(10, 6), dpi=100, is_save=False, save_name="draw_lines"):
    """
    绘制2d折线图，可以同时绘制多幅图
    :param x: 横坐标对应的值
    :param y: 纵坐标对应的值
    :param title: 每幅图的标题
    :param x_axis_names: 每幅图的横坐标名称
    :param y_axis_names: 每幅图的纵坐标名称
    :param fig_num: 需要绘制的子图数量
    :param subgraph_dis: 子图排布的方式，“22”表示以2x2的形式绘制子图
    :param figsize: 图片比例
    :param dpi: 分辨率
    :return:
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = []
    for i in range(fig_num):
        ax.append(fig.add_subplot(subgraph_dis * 10 + (i + 1)))
        ax[i].set(title=title[i], xlabel=x_axis_names[i], ylabel=y_axis_names[i])
        ax[i].plot(x[i, :], y[i, :])

    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    fig.tight_layout()
    # plt.show()
    if is_save:
        fig.savefig(f'./data_anylysis/{save_name}.pdf', bbox_inches='tight')  # dpi=dpi, format='svg',


def draw_scatter(x, y, title, x_axis_names, y_axis_names, fig_num=1, subgraph_dis=11
                 , figsize=(10, 6), dpi=100, marker='o'):
    """
        绘制2d散点图，可以同时绘制多幅图
        :param x: 横坐标对应的值
        :param y: 纵坐标对应的值
        :param title: 每幅图的标题
        :param x_axis_names: 每幅图的横坐标名称
        :param y_axis_names: 每幅图的纵坐标名称
        :param fig_num: 需要绘制的子图数量
        :param subgraph_dis: 子图排布的方式，“22”表示以2x2的形式绘制子图
        :param figsize: 图片比例
        :param dpi: 分辨率
        :param marker: 点的样式
        :return:
        """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = []
    for i in range(fig_num):
        ax.append(fig.add_subplot(subgraph_dis * 10 + (i + 1)))
        ax[i].set(title=title[i], xlabel=x_axis_names[i], ylabel=y_axis_names[i])
        ax[i].scatter(x[i, :], y[i, :], marker=marker)

    fig.tight_layout()
    plt.show()


def draw_line_scatter(line_x, line_y, scatter_x, scatter_y, title, x_axis_names, y_axis_names, fig_num=1,
                      subgraph_dis=11, figsize=(10, 6), dpi=100, marker='o', line_mean='line', scatter_mean='scatter',
                      loc='best', line_color='b', scatter_color='b', is_save=False, save_name="draw_lineline",
                      big_title=""):
    """
        绘制2d散点和折线的混合图，主要用来实现那种拟合之后与原数据比较的场景
        :param line_x: 折线图横坐标对应的值
        :param line_y: 折线图纵坐标对应的值
        :param scatter_x: 散点图横坐标对应的值
        :param scatter_y: 散点图纵坐标对应的值
        :param title: 每幅图的标题
        :param x_axis_names: 每幅图的横坐标名称
        :param y_axis_names: 每幅图的纵坐标名称
        :param fig_num: 需要绘制的子图数量
        :param subgraph_dis: 子图排布的方式，“22”表示以2x2的形式绘制子图
        :param figsize: 图片比例
        :param dpi: 分辨率
        :param marker: 点的样式
        :param line_mean: 折线图例的标签
        :param scatter_mean: 散点图例的标签
        :param loc: 图例位置
        :param line_color: 线条颜色
        :param scatter_color: 散点颜色
        :param is_save: 存储图像
        :return:
        """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = []
    for i in range(fig_num):
        ax.append(fig.add_subplot(subgraph_dis * 10 + (i + 1)))
        ax[i].set(title=title[i], xlabel=x_axis_names[i], ylabel=y_axis_names[i])
        ax[i].plot(line_x[i], line_y[i], label=line_mean, color=line_color)
        ax[i].legend(loc=loc)
        ax[i].scatter(scatter_x[i], scatter_y[i], alpha=0.8, marker=marker, label=scatter_mean, color=scatter_color)
        ax[i].legend(loc=loc)

    plt.suptitle(big_title, fontsize=20)
    fig.tight_layout()
    plt.show()

    if is_save:
        fig.savefig(f'data_anylysis/{save_name}.pdf', bbox_inches='tight')  # dpi=dpi, format='svg',


def draw_line_line(line_x1, line_y1, line_x2, line_y2, title, x_axis_names, y_axis_names, fig_num=1,
                   subgraph_dis=11, figsize=(10, 6), dpi=100, line1_mean='line1', line2_mean='line2',
                   loc='best', color1='k', color2='m', is_save=False, save_name="draw_lineline", big_title=""):
    """
        绘制折线和折线的混合图，主要用来实现那种拟合之后与原数据比较的场景
        :param line_x1: 折线图横坐标对应的值
        :param line_y1: 折线图纵坐标对应的值
        :param line_x2: 折线图横坐标对应的值
        :param line_y2: 折线图横坐标对应的值
        :param title: 每幅图的标题
        :param x_axis_names: 每幅图的横坐标名称
        :param y_axis_names: 每幅图的纵坐标名称
        :param fig_num: 需要绘制的子图数量
        :param subgraph_dis: 子图排布的方式，“22”表示以2x2的形式绘制子图
        :param figsize: 图片比例
        :param dpi: 分辨率
        :param line1_mean: 折线1图例的标签
        :param line2_mean: 折线2图例的标签
        :param loc: 图例位置
        :param color1: 折线图1的颜色
        :param color2: 折线图2的颜色
        :param is_save: 存储图像
        :return:
        """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.title(big_title, x=0.5, y=1.1, fontsize=20)
    ax = []
    for i in range(fig_num):
        ax.append(fig.add_subplot(subgraph_dis * 10 + (i + 1)))
        ax[i].set(title=title[i], xlabel=x_axis_names[i], ylabel=y_axis_names[i])
        ax[i].plot(line_x1[i], line_y1[i], label=line1_mean, c=color1)
        ax[i].legend(loc=loc)
        ax[i].plot(line_x2[i], line_y2[i], label=line2_mean, c=color2)
        ax[i].legend(loc=loc)

    fig.tight_layout()
    plt.show()
    if is_save:
        fig.savefig(f'data_anylysis/{save_name}.pdf', bbox_inches='tight')  # dpi=dpi, format='svg',


def draw_lines(x, y, title, x_axis_names, y_axis_names, line_num=1, figsize=(10, 6), dpi=100,
               line_mean=[], loc='best', color=[], x_intervals=1, is_save=False, save_as="draw_lines"):
    """
        绘制折线和折线的混合图，主要用来实现那种拟合之后与原数据比较的场景
        :param x: 折线图横坐标对应的值，列表
        :param y: 折线图纵坐标对应的值，列表
        :param title: 每幅图的标题
        :param x_axis_names: 每幅图的横坐标名称
        :param y_axis_names: 每幅图的纵坐标名称
        :param line_num: 需要绘制的折线数量
        :param figsize: 图片比例
        :param dpi: 分辨率
        :param line_mean: 折线图例的标签
        :param loc: 图例位置
        :param color: 折线图的颜色
        :return:
        """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.title(title)
    plt.xlabel(x_axis_names, fontsize=20)
    plt.ylabel(y_axis_names, fontsize=20)
    for i in range(line_num):
        temp_line_mean = 'Unknown'
        temp_color = 'k'
        if len(line_mean) > i:
            temp_line_mean = line_mean[i]
        if len(color) > i:
            temp_color = color[i]
        plt.plot(x[i], y[i], label=temp_line_mean, c=temp_color)
        plt.legend(loc=loc)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=18)
    # plt.show()
    if is_save:
        fig.savefig(f'{save_as}.pdf', bbox_inches='tight')  # dpi=dpi, format='svg',
        plt.close(fig)


def draw_hist(x, title, x_axis_names, y_axis_names, fig_num=1, subgraph_dis=11, figsize=(10, 6), dpi=100, bins=None,
              hist_range=None, density=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid',
              orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False):
    """
    绘制直方图
    :param x: 数据
    :param title: 图像名称
    :param x_axis_names: 横坐标名称
    :param y_axis_names: 纵坐标名称
    :param bins: 设置长条形的数目
    :param hist_range:指定直方图数据的上下界，默认包含绘图数据的最大值和最小值（范围）
    :param density:如果"True"，将y轴转化为密度刻度 默认为None
    :param weights:该参数可为每一个数据点设置权重
    :param cumulative:是否需要计算累计频数或频率 默认值False
    :param bottom:可以为直方图的每个条形添加基准线，默认为0
    :param histtype:{'bar', 'barstacked', 'step', 'stepfilled'}
        bar柱状形数据并排，默认值。
        barstacked在柱状形数据重叠并排（相同的在一起）
        step柱状形颜色不填充
        stepfilled填充的线性
    :param align:'mid' or 'left' or 'right' 设置条形边界值的对其方式，默认为mid，除此还有’left’和’right’
    :param orientation:{'vertical', 'horizontal'}设置直方图的摆放方向，默认为垂直方向vertical
    :param rwidth:设置直方图条形宽度的百分比
    :param log:是否需要对绘图数据进行log变换 默认值False
    :param color:设置直方图的填充色
    :param label:设置直方图的标签
    :param stacked:当有多个数据时，是否需要将直方图呈堆叠摆放，默认False水平摆放；
    :param fig_num:需要绘制的子图数量
    :param subgraph_dis:子图排布的方式，“22”表示以2x2的形式绘制子图
    :param figsize:图片比例
    :param dpi:分辨率
    :return:
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = []
    for i in range(fig_num):
        ax.append(fig.add_subplot(subgraph_dis * 10 + (i + 1)))
        ax[i].set(title=title[i], xlabel=x_axis_names[i], ylabel=y_axis_names[i])
        ax[i].hist(x[i], bins=bins, range=hist_range, density=density, weights=weights, cumulative=cumulative,
                   bottom=bottom, histtype=histtype, align=align, orientation=orientation, rwidth=rwidth, log=log,
                   color=color, label=label, stacked=stacked)

    fig.tight_layout()
    plt.show()


def draw_heatmap(x, title, x_axis_names, y_axis_names, figsize=(10, 6), dpi=100, num_vision=False, color='jet'
                 , is_save=False, save_name="draw_heatmap"):
    """
    绘制直方图
    :param x: 数据
    :param title: 图像名称
    :param x_axis_names: 横坐标名称
    :param y_axis_names: 纵坐标名称
    :param figsize:图片比例
    :param dpi:分辨率
    :param num_vision:是否在每个块上显示数值
    :param jet:热力图颜色
        autumn	红-橙-黄
        bone	黑-白，x线
        cool	青-洋红
        copper	黑-铜
        flag	红-白-蓝-黑
        gray	黑-白
        hot	    黑-红-黄-白
        hsv	    hsv颜色空间， 红-黄-绿-青-蓝-洋红-红
        inferno	黑-红-黄
        jet	    蓝-青-黄-红
        magma	黑-红-白
        pink	黑-粉-白
        plasma	绿-红-黄
        prism	红-黄-绿-蓝-紫-...-绿模式
        spring	洋红-黄
        summer	绿-黄
        viridis	蓝-绿-黄
        winter	蓝-绿
    :return:
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.xticks(np.arange(len(x_axis_names)), labels=x_axis_names,
               rotation=45, rotation_mode="anchor", ha="right")
    plt.yticks(np.arange(len(y_axis_names)), labels=y_axis_names)
    plt.title(title)

    if num_vision:
        for i in range(len(y_axis_names)):
            for j in range(len(x_axis_names)):
                text = plt.text(j, i, x[i, j], ha="center", va="center", color="k")

    plt.imshow(x, cmap=color)
    plt.colorbar()
    # plt.tight_layout()
    plt.show()
    if is_save:
        fig.savefig(f'data_anylysis/{save_name}.pdf', bbox_inches='tight')  # dpi=dpi, format='svg',


def draw_pie(value, labels, title, fig_num=1, subgraph_dis=11, figsize=(10, 6), dpi=100, explode=None, shadow=False,
             is_legend=False):
    """
    绘制饼图
    :param value: 数据
    :param labels: 标签
    :param title: 名称
    :param fig_num:需要绘制的子图数量
    :param subgraph_dis:子图排布的方式，“22”表示以2x2的形式绘制子图
    :param figsize:图片比例
    :param dpi:分辨率
    :param explode:分块是否膨胀
    :param shadow:阴影
    :param is_legend:是否将名称以图例的形式展示
    :return:
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)

    ax = []
    for i in range(fig_num):
        ax.append(fig.add_subplot(subgraph_dis * 10 + (i + 1)))
        ax[i].set(title=title[i])
        if is_legend:
            ax[i].pie(value[i], autopct='%1.2f%%', shadow=shadow, startangle=90,
                      explode=explode, pctdistance=1.12)
            ax[i].legend(labels=labels[i], loc='upper right')
        else:
            ax[i].pie(value[i], labels=labels[i], autopct='%1.1f%%', shadow=shadow)
        ax[i].axis('equal')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    tx = np.linspace(0, np.pi * 6)  # math.pi==scipy.pi==np.pi 都是圆周率π
    ty = np.sin(tx)
    tx = np.expand_dims(tx, 0).repeat(4, axis=0)
    ty = np.expand_dims(ty, 0).repeat(4, axis=0)
    t_title = ['one', 'two', 'three', 'four']
    t_x_axis = ['plot', 'plot', 'plot', 'plot']
    t_y_axis = ['plot', 'plot', 'plot', 'plot']

    # draw_line(tx, ty, t_title, t_x_axis, t_y_axis, 4, 22)

    # draw_scatter(tx, ty, t_title, t_x_axis, t_y_axis, 4, 22)

    draw_line_scatter(tx, ty, tx, ty, t_title, t_x_axis, t_y_axis, 4, 22)

    ty2 = np.copy(ty)
    ty2 += (np.random.random(ty2.size).reshape(ty2.shape) - 0.5) * 2 / 10

    # draw_line_line(tx, ty, tx, ty2, t_title, t_x_axis, t_y_axis, 4, 22)

    # draw_lines(tx[:3], np.vstack((ty[0, :], ty2[:2, :])), title=t_title[0], x_axis_names=t_x_axis[0],
    #            y_axis_names=t_y_axis[0], line_num=3, line_mean=['line1', 'line2'], color=['r', 'b'])

    # -----------------------------------------

    # tx = []
    # tx.append(np.random.randn(999, 4))
    # tx.append(np.random.randn(999, 3))
    # tx.append(np.random.randn(999, 2))
    # tx.append(np.random.randn(999, 1))
    # draw_hist(tx, t_title, t_x_axis, t_y_axis, fig_num=4, subgraph_dis=22)

    # -----------------------------------------

    # vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
    #               "potato", "wheat", "barley"]
    # farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
    #            "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
    #
    # harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
    #                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
    #                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
    #                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
    #                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
    #                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
    #                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
    #
    # draw_heatmap(harvest, title="plot", x_axis_names=farmers, y_axis_names=vegetables, num_vision=True)
    title = ['Pie', 'Pie', 'Pie', 'Pie']
    labels = [['apple', 'banana', 'watermelon', 'strawberry', 'else'],
              ['apple', 'banana', 'watermelon', 'strawberry', 'else'],
              ['apple', 'banana', 'watermelon', 'strawberry', 'else'],
              ['apple', 'banana', 'watermelon', 'strawberry', 'else']]
    sizes = [[10, 35, 25, 25, 5], [10, 35, 25, 25, 5], [10, 35, 25, 25, 5], [10, 35, 25, 25, 5]]
    explode = (0, 0.1, 0, 0, 0)

    draw_pie(sizes, labels, title, 4, 22, shadow=False, is_legend=True)
