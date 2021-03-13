#include <iostream>
#include <bits/stdc++.h>
using namespace std;


//递归的精髓，使用函数可认为已经算出结果只需最后一步

//并查集！！！！！
class Djset {
public:
    vector<int> parent;  // 记录节点的根
    vector<int> rank;  // 记录根节点的深度（用于优化）
    Djset(int n): parent(vector<int>(n)), rank(vector<int>(n)) {
        //初始化：每个结点的根节点均为自身
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }
    
    int find(int x) {
        // 压缩方式：直接指向根节点 （路径压缩）
        if (x != parent[x]) {
        	//直接将子结点变为根节点的孩子结点
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }
    
    void merge(int x, int y) {
        int rootx = find(x);
        int rooty = find(y);
        if (rootx != rooty) {
            // 按秩合并 将x作为阶数大的结点
            if (rank[rootx] < rank[rooty]) {
                swap(rootx, rooty);
            }
            //阶数小的y的根节点为x
            parent[rooty] = rootx;
            //更新x的阶数
            if (rank[rootx] == rank[rooty]) rank[rootx] += 1;
        }
    }
};

//移除最多的同行或同列石头
class Djset {
public:
    unordered_map<int, int> parent, rank;  // 记录节点的根
    int count;
    Djset(int n): count(0) {}

    int find(int x) {
        // 添加了一个新的集合，count+1
        if (parent.find(x) == parent.end()) {
            parent[x] = x;
            count++;
        }
        // 压缩方式：直接指向根节点
        if (x != parent[x]) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    void merge(int x, int y) {
        int rootx = find(x);
        int rooty = find(y);
        if (rootx != rooty) {
            if (rank[rootx] < rank[rooty]) {
                swap(rootx, rooty);
            }
            parent[rooty] = rootx;
            if (rank[rootx] == rank[rooty]) rank[rootx] += 1;
            count--;
        }
    }

    int get_count() {
        return count;
    }
};
class Solution {
public:
    int removeStones(vector<vector<int>>& stones) {
        int n = stones.size();
        Djset ds(n);
        for (auto e : stones) {
            //x和y的值可能冲突，所以这里我们将x加上10001
            ds.merge(e[0] + 10001, e[1]);
        }
        return n - ds.get_count();
    }
};

class Person
{
	friend ostream &operator<< (ostream &cout, Person &p);
	public:
		
		Person (int a, int b) : m_A(a), m_B(b) {}
		
	private:
	
		int m_A;
		int m_B;
};

void test01() 
{
	Person p(10, 20);
	cout << p << "hello world !!!" << endl;
}

ostream &operator<< (ostream &cout, Person &p)
{
	cout <<"m_A = " << p.m_A << endl;
	cout <<"m_B = " << p.m_B << endl; 
	return cout;
}


class Solution {
public:
	//滑动窗口 计算有多少种组合可以组成target
    vector<vector<int>> findContinuousSequence(int target) {
        vector<vector<int>> res;
        int i = 1, j = 1;
        int sum = 0;
        while (i <= target / 2) {
            if (sum < target) {
                // 右边界向右移动
                sum += j;
                j++;
            } else if (sum > target) {
                // 左边界向右移动
                sum -= i;
                i++;
            } else {
                // 记录结果
                vector<int> arr;
                for (int k = i; k < j; k++) {
                    arr.push_back(k);
                }
                res.push_back(arr);
                // 左边界向右移动
                sum -= i;
                i++;
            }
        }
        return res;
    }

    //滑动窗口 替换后的最长重复字符
    int characterReplacement(string s, int k) {
        int len = s.length();
        vector<int> count(26);
        int left = 0, right = 0, maxCount = 0;
        //int res = 0;
        while (right < len) {
            count[s[right] - 'A']++;
            maxCount = max(maxCount, count[s[right] - 'A']);
            right++;
            if (right - left > maxCount + k) {
                count[s[left] - 'A']--;
                left++;
            }
            //res = max(res, right - left);
        }
//由于maxCount的值只增不减，所以[left,right)能表示的最大范围也不会减少，所以最后[left,right)不是所求，但数值上相等。
        return right - left;
        //return res;   
    }

    
    //统计数组中出现为1的个数
    int singleNumber(vector<int>& nums) {
        int ans = 0;
        for(int i = 0; i < 32; ++i){
            int cnt = 0;
            for(int n : nums){
                // n & 1 << i 的值大于0即为真
                if(n & (1 << i)) cnt++;
            }
            // 构造只出现一次的那个数字，直接+= 即可
            if(cnt % 3 == 1) ans += (1 << i);
        }
        return ans;
    }

    //从上到下、从左到右递增查找是否有target值
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        if(matrix.empty() || matrix[0].empty()) return false;
        int m = matrix.size(), n = matrix[0].size();   //矩阵为m行，n列
        int row = m-1, col = 0;                        //从左下角开始寻找目标值
        while(row >= 0 && col <= n-1)
        {
            if(target > matrix[row][col])        ++col;
            else if(target < matrix[row][col])   --row;
            else if(target == matrix[row][col])  return true;
        }
        return false;
    }

    //滑动窗口的最大值
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> res;
        if (nums.empty())   return res;
        deque<int> d; //单调队列 保存从大到小的数据
        int i = 0, max = INT_MIN, j;
        for (i = 0; i < k; i++) {
            while (!d.empty() && d.back() < nums[i]) { //插入不单调 直接弹出
                d.pop_back();
            }
            d.push_back(nums[i]); //插入
        }
        res.push_back(d.front()); //插入最大值
        int len = nums.size();
        for (j = i - k; i < len; i++, j++) {
            if (nums[j] == d.front()) { //滑动窗口移动 如果恰为最大值，弹出
                d.pop_front();
            }
            while (!d.empty() && d.back() < nums[i]) {
                d.pop_back();
            }
            d.push_back(nums[i]);
            res.push_back(d.front());
        }
        return res;
    }

    /*有多少种翻译的可能
    	如果i-2与i-1的结果为翻译结果内
    		dp[i] = dp[i-1]*1 + dp[i-2]*1;
    	否则
			dp[i] = dp[i-1];
    					*/
    int translateNum(int num) {
        if(num < 10) return 1;
        //转换为字符串
        string s = to_string(num);
        int len = s.length();
        vector<int> dp(len + 1);
        dp[0] = 1; dp[1] = 1; 
        for(int i = 2; i < len + 1; ++i) {
            if(s[i-2] == '1' || (s[i-2] == '2' && s[i-1] <= '5')) {
                dp[i] = dp[i-2] + dp[i-1];
            }
            else {
                dp[i] = dp[i-1];
            }
        }
        return dp[len];
    }

    //约瑟夫环 0~n-1 
    int lastRemaining(int n, int m) {
    	//只有一个人时，自己时幸存者， f = 0;
        int f = 0;
        //2、3、...、n个人的迭代
        for (int i = 2; i <= n; i++){
            f = (m+f)%i;
        }
        return f;
    }

    //在字符串 s 中找出第一个只出现一次的字符（string的遍历）
    char firstUniqChar(string s) {
        unordered_map<char, int> dic;
        for(char& c : s)
            dic[c]++;
        for(char& c : s)
            if(dic[c] == 1) return c;
        return ' ';
    }

    /*
	// Definition for a Node.
	class Node {
	public:
	    int val;
	    Node* next;
	    Node* random;
	    
	    Node(int _val) {
	        val = _val;
	        next = NULL;
	        random = NULL;
	    }
	};
	*/
    //复杂链表的复制
    Node *copyRandomList(Node *head) {
        unordered_map<Node*, Node*> mp;
        Node *cur = head;
        while(cur != nullptr){
        	mp[cur] = new Node(cur->val);
        	cur = cur->next;
        }
        cur = head;
        while(cur != nullptr){
        	mp[cur]->next = mp[cur->next];
        	mp[cur]->random = mp[cur->random];
            cur = cur->next;
        }
        return mp[head];
    }


    //循环遍历：B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]
    vector<int> constructArr(vector<int>& a) {
        int n = a.size();
        vector<int> res(n);
        int left = 1;
        for(int i = 0; i < n; i++){
            res[i] = left;
            left *= a[i];
        }
        int right = 1;
        for(int i = n-1; i >= 0; i--){
            res[i] *= right;
            right *= a[i];
        }
        return res;
    }

    //字符串全排列：next_permutation
    vector<string> permutation(string s) {
		vector<string> res;
		//按照递增排序
		sort(s.begin(), s.end());
		do {
			res.push_back(s);
		} while (next_permutation(s.begin(), s.end()));
		return res;
	}

	//同上:回溯法
	vector<string> res;
    vector<string> permutation(string s) {
        dfs(0, s); // 从 s 的第一位开始排列起，所以传了 0
        return res;
    }
    void dfs(int pos, string& s){
        int len = s.length();
        if(pos == len-1){
            res.push_back(s); // 某条路从头排列到尾了，把这条路的结果输入 res
            return;
        }
        unordered_set<char> prune; 
        for(int i = pos; i < len; ++i){
            if( prune.find(s[i]) != prune.end() ){
                continue; // 如果是重复的元素，不要再排一次
            }
            prune.insert(s[i]);
            swap(s[i], s[pos]); // 每个诸侯都有暂时做天子（从 i 处到 pos 处）的机会
            dfs(pos+1, s); // 开始往下一层递进（当前诸侯需要寻找皇位（pos 处）继承人）
            swap(s[i], s[pos]); // 当前诸侯任期已满，从 pos 处回到 i 处
        }
    }


    //N个骰子点数 dp[i][j]为i个骰子点数为j的可能
    vector<double> dicesProbability(int n) {
        vector<vector<int>> dp(n+1, vector<int>(6*n + 1));
        dp[1][1] = 1; dp[1][2] = 1; dp[1][3] = 1;
        dp[1][4] = 1; dp[1][5] = 1; dp[1][6] = 1;

        for (int i = 2; i < n+1; ++i) {
            // j从i开始 , 因为j的最小值为i
            for (int j = i; j < 6*i+1; ++j) {
                for (int k = 1; k < 7; ++k) {
                    if (j-k > 0)
                        dp[i][j] += dp[i-1][j-k];
                }
            }
        }

        double sum = pow(6, n);
        vector<double> res;
        for (int i = n; i < 6 * n + 1; ++i) {
            // [n][n]~[n][6n]
            res.push_back(double(dp[n][i]) / sum);
        }
        return res;
    }


    //leetcode 220
    bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {
    	if (nums.empty()) return false;
        set<long> st;
        for (int i = 0; i < nums.size(); ++i) {
            //找大于等于s >= num[i] - t的数                        
            auto s = st.lower_bound((long)nums[i] - t);
            //如果找到并且这个数s <= nums[i] + t，返回true
            if (s != st.end() && *s <= (long)nums[i] + t) return true;
            st.insert(nums[i]);
            //保证i与j的差的绝对值小于等于k
            if (st.size() > k) {
                st.erase(nums[i - k]);
            }
        }
        return false;
    }

    //两数相加 用链表存储
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode *head = nullptr, *tail = nullptr;
        int carry = 0;
        while (l1 || l2) {
            int n1 = l1 ? l1->val: 0;
            int n2 = l2 ? l2->val: 0;
            int sum = n1 + n2 + carry;
            if (head == nullptr) {
                head = tail = new ListNode(sum % 10);
            } else {
                tail->next = new ListNode(sum % 10);
                tail = tail->next;
            }
            carry = sum / 10;
            if (l1) {
                l1 = l1->next;
            }
            if (l2) {
                l2 = l2->next;
            }
        }
        if (carry > 0) {
            tail->next = new ListNode(carry);
        }
        return head;
    }

    //每次只能从前后取， 如何取才能保证最大
    int maxScore(vector<int>& cardPoints, int k) {
    	int n = cardPoints.size();
    	//找到n-k个最小的，剩下的必然是最大的
    	int windowSize = n - k;
    	int sum = accumulate(cardPoints.begin(), cardPoints.begin() + windowSize, 0);
    	int minRes = sum;
    	for(int i = windowSize; i < n; ++i){
    		sum += cardPoints[i] - cardPoints[i - windowSize];
    		minRes = min(minRes, sum);
    	}
    	return accumulate(cardPoints.begin(), cardPoints.end(), 0) - minRes;
    }

    //乘积最大子数组
    int maxProduct(vector<int>& nums) {
    	int n = nums.size();
        int maxF = nums[0], minF = nums[0], ans = nums[0];
        for (int i = 1; i < n; ++i) {
            int mx = maxF, mn = minF;
            maxF = max(mx * nums[i], max(nums[i], mn * nums[i]));
            minF = min(mn * nums[i], min(nums[i], mx * nums[i]));
            ans = max(maxF, ans);
        }
        return ans;
    }


    //K个一组翻转链表
    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode* hair = new ListNode(-1);
        hair->next = head;
        ListNode* L = hair;
        int length = 0;
        while(hair->next != nullptr){
            ++length;
            hair = hair->next;
        }
        hair = L;
        int count = length / k;
        for (int i = 0; i < count; ++i) {
            for (int j = 1; j < k; j++) {
                ListNode* temp = hair->next;
                hair->next = head->next;
                head->next = head->next->next;
                hair->next->next = temp;
            }
            hair = head;
            head = head->next;
        }
        return L->next;
    }

    //三数之和为0
    vector<vector<int>> threeSum(vector<int>& nums) {
     	vector<vector<int>> res;
     	int n = nums.size();
     	sort(nums.begin(), nums.end());
     	for(int i = 0; i < n; i++){
     		if(nums[i] > 0)
     			break;
     		if(i > 0 && nums[i] == nums[i-1]) continue;
     		int j = i+1;
     		int k = n-1;
     		while(j < k){
     			int sum = nums[i]+nums[j]+nums[k];
     			if(sum == 0){
     				res.push_back({nums[i], nums[j], nums[k]});
     				while(j < k && nums[j] == nums[j+1]) j++;
     				while(j < k && nums[k] == nums[k-1]) k--;
     				j++;
     				k--;
     			}
     			else if (sum < 0)
     				j++;
     			else
     				k--;
     		}
     	}
     	return res;
    }

    //爱生气的老板
    int maxSatisfied(vector<int>& customers, vector<int>& grumpy, int X) {
        int ans = 0, cur = 0;
        int n = customers.size();
        //不生气一定为能满意的
        for(int i = 0; i < n; i++){
            if(grumpy[i] == 0)
                ans += customers[i];
        }
        //长度为X的滑动窗口能增长的顾客数
        for(int i = 0; i < X; i++){
            if(grumpy[i] == 1)
                cur += customers[i];
        }
        int res = cur;
        //窗口移动 比较最大能增长多少顾客
        for(int i = X; i < n; i++){
            if(grumpy[i] == 1)
                cur += customers[i];
            if(grumpy[i-X] == 1)
                cur -= customers[i-X];
            res = max(res, cur);
        }
        return ans + res;
    }

    //前序和中序建二叉树
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    	return constructTree(preorder.begin(), preorder.end(), inorder.begin(), inorder.end());
    }
    TreeNode* constructTree(vector<int>::iterator preBegin, vector<int>::iterator preEnd,
    						vector<int>::iterator inBegin, vector<int>::iterator inEnd) {
    	if(preBegin == preEnd)
    		return nullptr;
    	TreeNode* cur = new TreeNode(*preBegin);
    	auto root_pos = find(inBegin, inEnd, *preBegin);
    	cur->left = constructTree(preBegin+1, preBegin+1+(root_pos-inBegin), inBegin, root_pos+1);
    	cur->right = constructTree(preBegin+1+(root_pos-inBegin), preEnd, root_pos+1, inEnd);
    	return cur; 
    }


    //最近公共祖先
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    	if(root == nullptr || root == p || root == q)
    		return root;
    	TreeNode* left = lowestCommonAncestor(root->left, p, q);
    	TreeNode* right = lowestCommonAncestor(root->right, p, q);
    	if(left && right)
    		return root;
    	return left ? left : right;
    }


    //接雨水
    int trap(vector<int>& height) {
    	int n = height.size();
    	int res = 0;
    	if(n < 3)
    		return res;
    	vector<int>leftMax(n), rightMax(n);
    	leftMax[0] = height[0];
    	for(int i = 1; i < n-1; i++)
    		leftMax[i] = max(height[i], leftMax[i-1]);
    	rightMax[n-1] = height[n-1];
    	for(int i = n-2; i > 0; i--)
    		rightMax[i] = max(height[i], rightMax[i+1]);
    	for(int i = 1; i < n-1; i++)
    		res += min(leftMax[i], rightMax[i]) - height[i];
    	return res;
    }


    //树的路径总和II
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
    	vector<vector<int>> res;
    	if(root == nullptr)
    		return res;
    	vector<int> path;
    	recur(root, targetSum, res, path);
    	return res;
    }

    void recur(TreeNode* root, int targetSum, vector<vector<int>>& res, vector<int>& path){
    	path.push_back(root->val);
    	if(root->val == targetSum && root->right == nullptr && root->left == nullptr){
    		res.push_back(path);
            path.pop_back();
    		return;
    	}
    	targetSum -= root->val;
    	if(root->left)
    		recur(root->left, targetSum, res, path);
    	if(root->right)
    		recur(root->right, targetSum, res, path);
    	targetSum += root->val;
    	path.pop_back();
    }

    //二叉树中的最大路径和
    int val;
    int maxPathSum(TreeNode* root) {
    	if(root == nullptr) return 0;
    	val = INT_MIN;
        maxGain(root);
    	return val;
    }

    int maxGain(TreeNode* root){
    	if(root == nullptr) return 0;
    	int leftVal = max(0, maxGain(root->left));
    	int rightVal = max(0, maxGain(root->right));
    	val = max(val, leftVal + rightVal + root->val);
    	return root->val + max(leftVal, rightVal);
    }

    //二叉树的直径
    int maxCount;
    int diameterOfBinaryTree(TreeNode* root) {
    	maxCount = 1;
    	maxGain(root);
    	return maxCount - 1;
    }
    int maxGain(TreeNode* root){
    	if(root == nullptr)	return 0;
    	int left = maxGain(root->left);
    	int right = maxGain(root->right);
    	maxCount = max(maxCount, left + right + 1);
    	return max(left, right) + 1;
    }
    
    //零钱兑换  贪心+剪枝
    void coinChange(vector<int>& coins, int amount, int cur, int count, int& res){
    	if(amount == 0){
    		res = min(res, count);
    		return;
    	}
    	if(cur == coins.size())
    		return;
    	for(int k = amount / coins[cur]; k >= 0 && count + k < res; k--){
    		coinChange(coins, amount - k*coins[cur], cur+1, count+k, res);
    	}
    }

    int coinChange(vector<int>& coins, int amount) {
    	sort(coins.rbegin(), coins.rend());
    	int res = INT_MAX;
    	coinChange(coins, amount, 0, 0, res);
    	return res == INT_MAX ? -1 : res;
    }

    //两数相加 ：链表：反向存储
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    	ListNode *head = nullptr, *tail = nullptr;
    	int carry = 0;
    	while(l1 || l2){
    		int x = l1 ? l1->val : 0;
    		int y = l2 ? l2->val : 0;
    		int sum = x + y + carry;
    		if(head == nullptr){
    			head = tail = new ListNode(sum % 10);
    		} else {
    			tail->next = new ListNode(sum % 10);
    			tail = tail->next;
    		}
    		carry = sum / 10;
    		if(l1)
    			l1 = l1->next;
    		if(l2)
    			l2 = l2->next;
    	}
    	if(carry > 0){
    		tail->next = new ListNode(carry);
    	}
    	return head;
    }

    //不同的二叉搜索树 dp
    int numTrees(int n) {
    	vector<int> dp(n+1, 0);
    	dp[0] = 1; dp[1] = 1;
    	for(int i = 2; i <= n; i++){
    		for(int j = 1; j <= i; j++)
    			dp[i] += dp[j-1] * dp[i-j];
    	}
    	return dp[n];
    }

    //下一个更大的元素III
    int nextGreaterElement(int n) {
    	string src = to_string(n);
    	int len = src.length();
    	int i = len-2;
    	while(i >= 0 && src[i] >= src[i+1])
    		i--;
    	if(i < 0 || src[i] >= src[i+1])
    		return -1;
    	int j = i;
    	while(src[j+1] > src[i])
    		j++;
    	swap(src[i], src[j]);
    	sort(src.begin() + i + 1, src.end());
    	long long ans = stoll(src);
    	if(ans > INT_MAX)
    		return -1;
    	return ans;
    }

    //最长回文子串 : 中心扩展法
    pair<int, int> expendAroundCenter(string& s, int left, int right){
    	while(left >= 0 && right < s.length() && s[left] == s[right]){
    		left--; right++;
    	}
    	return make_pair(left+1, right-1);
    }
    string longestPalindrome(string s) {
    	int n = s.length();
    	int start = 0, end = 0;
    	for(int i = 0; i < n; i++){
    		auto odd = expendAroundCenter(s, i, i);
    		auto even = expendAroundCenter(s, i, i+1);
    		if (odd.second - odd.first > end - start) {
                start = odd.first;
                end = odd.second;
            }
            if (even.second - even.first > end - start) {
                start = even.first;
                end = even.second;
            }
    	}
    	return s.substr(start, end - start + 1);
    }

    //全排列
    void backtrack(vector<vector<int>>& res, vector<int>& nums, vector<int>& cur, vector<bool>& vis){
        if(cur.size() == nums.size()){
            res.push_back(cur);
            return;
        }
        for(int i = 0; i < nums.size(); ++i){
            if(vis[i])  continue;
            vis[i] = true;
            cur.push_back(nums[i]);
            backtrack(res, nums, cur, vis);
            cur.pop_back();
            vis[i] = false;
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int> > res;
        vector<int> cur;
        vector<bool> vis(nums.size(), false);
        backtrack(res, nums, cur, vis);
        return res;
    }

    //字符串解码
    string decode(string& s, int& pos){
    	string res;
    	int num = 0;
    	while(pos < s.size()){
    		if(s[pos] >= '0' && s[pos] <= '9'){
    			num = num * 10 +  s[pos] - '0';
    		} 
            else if(s[pos] == '['){
    			string temp = decode(s, ++pos);
    			while(num){
    				num--;
    				res += temp;
    			}
    		} 
            else if(s[pos] == ']') break;
    		else res += s[pos];
    		pos++;
    	}
    	return res;
    }

    string decodeString(string s) {
        int pos = 0;
    	return decode(s, pos);
    }

    //二叉树的完全性检验
    bool isCompleteTree(TreeNode* root) {
    	queue<TreeNode*> q;
    	q.push(root);
    	bool flag = false;
    	while(q.size()){
    		auto cur = q.front();
    		q.pop();
    		if(cur == nullptr){
    			flag = true;
    			continue;
    		}
    		if(flag) return false;
    		q.push(cur->left);
    		q.push(cur->right);
    	}
    	return true;
    }

    //括号生成
    vector<string> generateParenthesis(int n) {
    	vector<string> res;
    	dfs(res, "", n, 0, 0);
    	return res;
    }
    void dfs(vector<string>& res, string path, int n, int left, int right){
    	if(left < right || left > n) return;
    	if(left == n && right == n){
    		res.push_back(path);
    		return;
    	}
    	dfs(res, path+'(', n, left+1, right);
    	dfs(res, path+')', n, left, right+1);
    }

    //两数相加II   (PS:对于逆序处理应该首先想到栈)
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        stack<int> s1, s2;
        while(l1){
            s1.push(l1->val);
            l1 = l1->next;
        }
        while(l2){
            s2.push(l2->val);
            l2 = l2->next;
        }
        int carry = 0;
        ListNode *head = new ListNode(-1);
        while(s1.size() || s2.size()){
            int x = s1.size() ? s1.top() : 0;
            int y = s2.size() ? s2.top() : 0;
            int sum = x + y + carry;
            ListNode *p = new ListNode(sum%10);
            p->next = head->next;
            head->next = p;
            carry = sum/10;
            if(s1.size())
                s1.pop();
            if(s2.size())
                s2.pop();
        }
        if(carry){
            ListNode *p = new ListNode(carry);
            p->next = head->next;
            head->next = p;
        }
        return head->next;
    }

    //用 Rand7() 实现 Rand10()
    int rand10() {
    	int col, row, idx;
    	do{
    		col = Rand7();
    		row = Rand7();
    		//直接乘 得不到所有数据 如11 、13等  [1,7] + [0,6]*7 = [1,49] 
    		idx = col + (row - 1) * 7;
    	} while(idx > 40);
    	return 1 + idx % 10;
    }

    //链表输入 
	 struct ListNode {
	     int val;
	     ListNode *next;
	     ListNode(int x) : val(x), next(NULL) {}
	 };
	 int n;
	 ListNode *head = NULL;
	 while(cin>>n){
	 	if(head == NULL){
	 		head = new ListNode(n);
	 		ListNode *p = head;
	 	}
	 	else{
	 		head->next = new ListNode(n);
	 		head = head->next;
	 	}
	 }
	 head = p;
};


class LRUCache {
public:
    LRUCache(int capacity){
    	m_capacity = capacity;
    }

    int get(int key){
    	if(hashtable.count(key) == 0) 
    		return -1;
    	else{
    		auto iter = hashtable[key];
    		cache.splice(cache.begin(), cache, iter);
    		hashtable[key] = cache.begin();
    		//begin()返回的是迭代器（指针）， 用->而不是.
    		return cache.begin()->second;
    	}
    }

    void put(int key, int value){
    	if(hashtable.count(key) == 0){
    		if(hashtable.size() == m_capacity){
    			//back()返回的是引用，只能用. 不能用->
    			hashtable.erase(cache.back().first);
    			cache.pop_back();
    		}
    		cache.push_front(make_pair(key, value));
    		hashtable[key] = cache.begin();
    	}
    	else{
    		auto iter = hashtable[key];
    		cache.splice(cache.begin(), cache, iter);
    		hashtable[key] = cache.begin();
    		cache.begin()->second = value;
    	}
    }

    int m_capacity;
    unordered_map<int, list<pair<int, int>>::iterator > hashtable;
    list<pair<int, int>> cache;
};


class NumMatrix {
public:
    vector<vector<int>> dp;

    NumMatrix(vector<vector<int>>& matrix) {
        if(matrix.size() == 0 || matrix[0].size() == 0)
        	return;
        int m = matrix.size(), n = matrix[0].size();
        dp.resize(m+1, vector<int>(n+1));
        for(int i = 1; i <= m; i++){
        	for(int j = 1; j <= n; j++)
        		dp[i][j] = dp[i][j-1] + dp[i-1][j] + matrix[i-1][j-1] - dp[i-1][j-1];
        }
    }

    int sumRegion(int row1, int col1, int row2, int col2) {
    	return dp[row2+1][col2+1] - dp[row2+1][col1] - dp[row1][col2+1] + dp[row1][col1];
    }
};
