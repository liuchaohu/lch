# powershell出现问题
每次打开powershell的时候，都会出现一段红字，大概内容是
`无法加载文件 C:\Users\liuc\Documents\WindowsPowerShell\profile.ps1`
- 解决办法是：
1. 以管理员身份打开powershell
2. 输入：`get-ExecutionPolicy`，显示：Restricted,这表示状态是禁止的
3. 这时输入:`set-ExecutionPolicy RemoteSigned` 就可以正常运行Python文件了
