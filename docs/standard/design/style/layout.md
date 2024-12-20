# 布局

页面布局是整个软件视觉设计的基础。在典型的中后台视觉体系中定义布局系统，我们建议从以下几个方面出发：
1. [标定画布尺寸](#标定画布尺寸)
2. [适配方案](#适配方案)
3. [网格单位](#网格单位)
4. [常用模度](#常用模度)

::: tip 🔔 TIP
活字格的目标用户为B端用户，所以讨论的主要场景为企业应用常见的中后台场景。如果您有C端场景，在适配方案上请做适当的调整。当然，设计背后的理念在多个场景中都是适用的。
:::

## 标定画布尺寸

为了尽可能在多端展现统一的效果，也为了团队开发时，减少沟通与理解的成本，画布的尺寸标定是必要的。活字格默认的画布尺寸为：

|  终端  |     尺寸     |    单位    |
|:----:|:----------:|:--------:|
|  PC  | 1200 × 900 | px - 像素  |
| 移动端  | 360 × 560  | px - 像素  |

在实际的开发过程中，我们应当考虑应用最终要访问的终端尺寸大小，其取决于终端屏幕的分辨率。一般来说，对于 web 网站的标准页面，页面高度不做任何限制，页面宽度可参考如下：

| 终端  |    尺寸范围    |    单位    |
|:---:|:----------:|:--------:|
| PC  | 950 ~ 1200 | px - 像素  |

当然，具体尺寸应当根据项目、客户需求以及用户群体决定。 画布大小与页面设计应当按照最小尺寸设计，屏幕尺寸超出画布的部分可以通过「范围模式」来响应式填充。

::: tip 🔔 TIP
移动端的尺寸在高精度标定下，还需要取决于手机品牌、操作系统、屏幕分辨率等多个因素，如果您有特定场景的需要，可参考第三方资源。
:::

此外，除了基础页面的画布尺寸标定外，还应当考虑弹窗、结果页等辅助页面的基础尺寸大小。

## 适配方案

对于布局的适配方案，建议使用「母版页」的特性来实现。

**左右布局**

常被用于左右布局的设计方案中，最常见的做法为将左边的导航栏固定，右边的「页面占位区」去进行动态缩放。

![左右布局示例](../../images/design-layout-adpat-horizontal.png "左右布局示例")

**上下布局**

常被用于上下布局的设计方案中，常见的作坊是将顶部的导航栏固定，下方的「页面占位区」作为工作区进行动态缩放。建议在设计时考虑对于两边留白的最小值定义。

![上下布局示例](../../images/design-layout-adpat-vertical.png "上下布局示例")

**L型布局**

对于复杂的中后台门户界面或后端配置台中，单一的菜单已经无法充分体现业务的维度，因此，我们会考虑将顶部导航与侧边导航进行组合，必要时可以选择标签页和打开标签页命令协同使用。

![L型布局示例](../../images/design-layout-adpat-L.png "L型布局示例")

::: tip 🔔 TIP
关于L型布局的样例，可以参考Demo工程：[统一门户管理框架](https://marketplace.grapecity.com.cn/ApplicationDetails?productID=SP2212230001&productDetailID=D2301130010&tabName=Tabs_detail)
:::

## 网格单位

活字格的页面设计是依托于网格体系来实现的。

对于一个单位的网格，默认尺寸为 20 × 20 像素。活字格没有采用编码 UI 框架的 24 栅格体系，而是使用类 Excel 的行列机制，将构建栅格体系的能力交给用户。因此，活字格的基础布局方式为 Grid 布局。

::: tip 🔔 TIP
如果在您的页面设计中存在 Flex 布局的需要，您可以使用图文列表，或者在单元格区域中，自定义其 CSS `display: flex` 即可。
:::

对于活字格的网格行列使用，请查阅 [页面流式布局](https://help.grapecity.com.cn/pages/viewpage.action?pageId=80952477)。

::: info 📍 INFO
本章节仅为大家介绍活字格网格的基础概念。了解这些知识，会为您下一节：[常用模度](#常用模度) 的学习提供理论基础。
:::

## 常用模度

模度，一个用来定义 UI 布局空间决策的概念。简单来说，就是 UI 布局设计中的关于尺寸比例的定义。

目前主流的设计标准符合“ 8 像素规则”，即布局中任何元素其自定义的长、宽以及间隙，都应该是 8 的倍数。

【<font color="#1677FF">推荐</font>】在活字格中，我们建议遵循 8 像素原则，提取一组适用于团队设计风格的模度数组，可以在一定程度上帮助我们更好的实现软件应用在布局空间上的一致性。 以下两组模度借鉴了主流的 UI 框架，可以作为您项目中的参考。

<table>
	<tr>
	    <td>8</td>
	    <td>12</td>
      <td style="background-color: #1677FF; color: white">20</td>
	    <td>32</td>
	    <td>48</td>
	    <td>80</td>
      <td>128</td>
	    <td>208</td>
	    <td>336</td>
	    <td>552</td>
      <td>896</td>
	    <td>1440</td>
	</tr>
	<tr>
      <td>4</td>
	    <td>16</td>
      <td>24</td>
	    <td>40</td>
	    <td>64</td>
	    <td>104</td>
      <td>168</td>
	    <td>272</td>
	    <td>440</td>
	    <td>720</td>
      <td>1152</td>
	    <td>1920</td>
	</tr>
</table>

::: info 📍 INFO
每一行表示一组模度。其中，蓝色背景的 `20` 为活字格默认的模度。
:::

