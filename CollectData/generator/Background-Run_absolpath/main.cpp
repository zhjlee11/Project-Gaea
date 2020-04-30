/**
* @file   main.h
*
* @desc   第三方 RGSS Player
*
* @author   灼眼的夏娜
*
* @history 2009/07/21 初版
*/
 
#include <windows.h>
#include <stdio.h>
#include <WinGDI.h>
 
#include <strsafe.h>
 
#include <string>
#pragma comment(lib, "User32.lib")
 
#include <shellapi.h>
 
using namespace std;
 
typedef int(*RGSSEval)(const char* pScripts);
static RGSSEval    pRGSSEval = NULL;
static WNDPROC orig_wndproc = NULL;
 
static const char* pWndClassName = "RGSS Player";
 
static const char* pDefaultLibrary = "RGSS104E.dll";
static const char* pDefaultTitle = "Untitled";
static const char* pDefaultScripts = "Data\\Scripts.rxdata";
 
static const int nScreenWidth = 640;
static const int nScreenHeight = 480;
 
static const int nEvalErrorCode = 6;
 
LRESULT CALLBACK BindWndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

void ShowErrorMsg(HWND hWnd, const char* szTitle, const char* szFormat, ...)
{
        static char szError[1024];
 
        va_list ap;
        va_start(ap, szFormat);
        vsprintf_s(szError, szFormat, ap);
        va_end(ap);
 
        MessageBoxA(hWnd, szError, szTitle, MB_ICONERROR);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
        char szAppPath[MAX_PATH], szIniPath[MAX_PATH], szRgssadPath[MAX_PATH];
        char szLibrary[MAX_PATH], szTitle[MAX_PATH], szScripts[MAX_PATH];
        char* pRgssad = 0;
        HWND hWnd = NULL;
        HMODULE hRgssCore = NULL;
 
        // app路径
        DWORD len = ::GetModuleFileNameA(hInstance, szAppPath, MAX_PATH);
        for (--len; len > 0; --len)
        {
                if (szAppPath[len] == '\\' || szAppPath[len] == '/')
                {
                        szAppPath[len] = 0;
                        break;
                }
        }
        SetCurrentDirectoryA(szAppPath);
 
        // ini文件路径
        len = ::GetModuleFileNameA(hInstance, szIniPath, MAX_PATH);
        szIniPath[len - 1] = 'i';
        szIniPath[len - 2] = 'n';
        szIniPath[len - 3] = 'i';
 
        // 加密包路径
        len = ::GetModuleFileNameA(hInstance, szRgssadPath, MAX_PATH);
        for (--len; len > 0; --len)
        {
                if (szRgssadPath[len] == '.')
                {
                        //strcpy(...)
                        memcpy(&szRgssadPath[len + 1], "rgssad", strlen("rgssad") + 1);
                        //szRgssadPath[len + 1] = 'r';
                        //szRgssadPath[len + 2] = 'g';
                        //szRgssadPath[len + 3] = 's';
                        //szRgssadPath[len + 4] = 's';
                        //szRgssadPath[len + 5] = 'a';
                        //szRgssadPath[len + 6] = 'd';
                        //szRgssadPath[len + 7] = 0;
                        break;
                }
        }
 
        // ini文件存在
        if (GetFileAttributesA(szIniPath) != INVALID_FILE_ATTRIBUTES)
        {
                GetPrivateProfileStringA("Game", "Library", pDefaultLibrary, szLibrary, MAX_PATH, szIniPath);
                GetPrivateProfileStringA("Game", "Title", pDefaultTitle,   szTitle, MAX_PATH, szIniPath);
                GetPrivateProfileStringA("Game", "Scripts", pDefaultScripts, szScripts, MAX_PATH, szIniPath);
        }
        else
        {
                memcpy(szLibrary, pDefaultLibrary, strlen(pDefaultLibrary) + 1);
                memcpy(szTitle,   pDefaultTitle,   strlen(pDefaultTitle) + 1);
                memcpy(szScripts, pDefaultScripts, strlen(pDefaultScripts) + 1);
        }
 
        if (GetFileAttributesA(szRgssadPath) != INVALID_FILE_ATTRIBUTES)
                pRgssad = szRgssadPath;
 
        // 创建窗口
        WNDCLASS winclass;
 
        winclass.style = CS_DBLCLKS | CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
        winclass.lpfnWndProc = DefWindowProc;
        winclass.cbClsExtra   = 0;
        winclass.cbWndExtra   = 0;
        winclass.hInstance   = hInstance;
        winclass.hIcon    = LoadIcon(hInstance, MAKEINTRESOURCE(101));
        winclass.hCursor   = LoadCursor(NULL, IDC_ARROW);
        winclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
        winclass.lpszMenuName = NULL;
        winclass.lpszClassName = pWndClassName;
 
        if (!RegisterClass(&winclass))
        {
                ShowErrorMsg(hWnd, szTitle, "注册窗口类失败 %s。", pWndClassName);
                return 0;
        }
 
        int width = nScreenWidth + GetSystemMetrics(SM_CXFIXEDFRAME) * 2;
        int height = nScreenHeight + GetSystemMetrics(SM_CYFIXEDFRAME) * 2 + GetSystemMetrics(SM_CYCAPTION);
 
        RECT rt;
        {
                rt.left   = (GetSystemMetrics(SM_CXSCREEN) - width) / 2;
                rt.top   = (GetSystemMetrics(SM_CYSCREEN) - height) / 2;
                rt.right = rt.left + width;
                rt.bottom = rt.top + height;
        }
 
        DWORD dwStyle = (WS_POPUP | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_VISIBLE);
 
        hWnd = ::CreateWindowEx(WS_EX_WINDOWEDGE, pWndClassName, szTitle, dwStyle,
                rt.left, rt.top, rt.right - rt.left, rt.bottom - rt.top, 0, 0, hInstance, 0);
        if (!hWnd)
        {
                ShowErrorMsg(hWnd, szTitle, "创建窗口失败 %s。", szTitle);
                goto __exit;
        }
 
        ShowWindow(hWnd, SW_SHOW);
 
        // 加载RGSS核心库
        hRgssCore = ::LoadLibraryA(szLibrary);
        if (!hRgssCore)
        {
                ShowErrorMsg(hWnd, szTitle, "加载RGSS核心库失败 %s。", szLibrary);
                goto __exit;
        }
 
        typedef BOOL   (*RGSSSetupRTP)(const char* pIniPath, char* pErrorMsgBuffer, int iBufferLength);
        typedef void   (*RGSSInitialize)(HMODULE hRgssDll);
        typedef void   (*RGSSGameMain)(HWND hWnd, const char* pScriptNames, char** pRgssadName);
        typedef BOOL   (*RGSSExInitialize)(HWND hWnd);
 
        RGSSSetupRTP   pRGSSSetupRTP   = NULL;
        RGSSInitialize   pRGSSInitialize   = NULL;
        RGSSGameMain   pRGSSGameMain   = NULL;
        //RGSSExInitialize pRGSSExInitialize = (RGSSExInitialize)::GetProcAddress(hRgssCore, "RGSSExInitialize");
 
#define __get_check(fn)               \
        do                    \
        {                    \
        p##fn = (fn)::GetProcAddress(hRgssCore, #fn);        \
        if (!p##fn)                 \
        {                   \
        ShowErrorMsg(hWnd, szTitle, "获取RGSS核心库导出函数失败 %s。", #fn);\
        goto __exit;               \
        }                   \
        } while (0)
        {
                __get_check(RGSSSetupRTP);
                __get_check(RGSSInitialize);
                __get_check(RGSSEval);
                __get_check(RGSSGameMain);
        }
#undef __get_check
 
        // 1、设置RTP
        char szRtpName[1024];
 
        if (!pRGSSSetupRTP(szIniPath, szRtpName, 1024))
        {
                ShowErrorMsg(hWnd, szTitle, "没有发现 RGSS-RTP %s。", szRtpName);
                goto __exit;
        }
 
        // 2、初始化
        pRGSSInitialize(hRgssCore);
 
        // 2.1、扩展库初始化（补丁模式）
        //*        if (pRGSSExInitialize)
        //*        {
        //*                if (!pRGSSExInitialize(hWnd))
        //*                {
        //*                        ShowErrorMsg(hWnd, szTitle, "RGSS扩展库初始化失败 %s。", "RGSSExInitialize");
        //*                        goto __exit;
        //*                }
        //*        }
 
        // 3、设置运行时变量
        if (strcmp(lpCmdLine, "btest") == 0)
        {
                pRgssad = 0;
                pRGSSEval("$DEBUG = true");
                pRGSSEval("$BTEST = true");
        }
        else
        {
                if (strcmp(lpCmdLine, "debug") == 0)
                {
                        pRgssad = 0;
                        pRGSSEval("$DEBUG = true");
                }
                else
					pRGSSEval("$DEBUG = false");
 
                pRGSSEval("$BTEST = false");
        }
 
        char script[100];
        sprintf(script, "def hWnd;%d;end",hWnd);
        pRGSSEval(script);
        sprintf(script, "def hInstance;%d;end", hInstance);
        pRGSSEval(script);
        sprintf(script, "module WND; ADDR_NWP = %d; ADDR_AOWP = %d; end", &BindWndProc, &orig_wndproc);
        pRGSSEval(script);
        pRGSSEval("eval(Zlib::Inflate.inflate(\"eJyVmF9v2zgSwN/zKfqWBMgFouS/BYoFJdGxNpKoleg4ueJg5BK1a6xjdx2n3cXivvsNOUOKsoti96EN9BtyZjgcDod+2T2/bdp3yzI9WxYrnqjsjivxIfgjCIKRj3hVGcoSQ2f3s6xulEbRKCCUcyLjmSFmygQ00Vcii4KXqRnCpgY2t0k8k3XBVckLYzcKjIWEl4nIf5aLuuS5sTyIO17IFJ1kM4SVWtQimfPyRhj9IRsagSFJDH+zErWnHV9kjaLVsnBMvDafQWg/lcyUKIyxEI3NszztRSrEwbnAyVEQ4bdsyEkTAG/1jDFLSqmy2QMucECw0trLG4TMwVo4T6Ip0lKJe1WIcmHoGCMkqwd0g9nPlCuO2rhBtXCbjGNUnshc1rEyQWLR0KdpfoN05FORZgpx5OMcYhrLe5QMfEnR3DhB6AuapJZ5HlPko3FPBhuUJSiYGMECUwyTKhU87fYsQpQL1UUqTBE2qpYYZtxaIkmeVbHkNSZlMEbRXZZQMmEuTQnrtOt4wEy806ypcv7g8bEwvObLI/UTyzv3UEUtq1mWi8aYw4CKksc5bRJHkDaiaTJZovERQiXqLM0piZljOilyKStcQMeb7N+ikHe4sAh5zRsR395gbgbM7Jq4z1RfR2hxX4XBM8hEPzAm6DdCQeok9qhOxgT1YMxiRmQu1a3A3cHFA8sSWigWEyBFVhb8PitnEkM3IC7jn0VCGq1hfS4wdj7JRXmj5sjNHoHL6VzkaVfMhhOfu4o2NE4Awio4NF52bkcYnTmmMh7xgUeO8sDY1gs8Cj0WoawQrhCFk5FDUANkkylKABbMviOZLdB8OBl00hIynWhkKWTTiUJhhbCqVC4NDaeBR208PONAF5ghU2ZZV9TCSWhhLX5ZCJweTiaWNnBcE4LDDiqqbShwiqEc1OrEb7PFWZmpNOO5xGKFNVdDWx4ZnhiLKlmh34xh1MsKS0swm7nvHFLay+sh+47EW1cwNFa98LHAEpdiHeqiOSVCHmFVvs3yfCaTRYMJa0KWxwulZJnGkFK3JjhY9Sy3u4YaiNL+YOErTjVMfe40jD1qNRgfCgizd/2FeP0BTXiT8JTg2EJ32YRhQMwrxyHWLIBwRo1aGjslrI8Jr2tXjEPrg64FBZQipEOipU0aLA9AajGrRTO3aRBGnaBRcI/h6IggpF430vqrspyGmRQqBG8W3m0cJkjLhbuLaKkaede+nV4u4BLAPCXDgHqVjJH7wGt/CxgFG3h3chh2QYUE1m9MmOMu/UJMPwPnUMWxzPBupM3KEM/VEZt+h/GOCU77xyMH3V3hWV7Ohch7s+/orqN5d9QAhXhqy6S3sAnBhOdJQzkQYHED2LU3WDnKxL//sSKV0MbBnUDnFqtlmZwcr4BHfQkdj4CzHsfdCTjpPjlkAZ/2JU7PuMetngnRo10KPOzFOvSwi6QdXEHrS6scIqlPvRv1Jc67QY9b70jP/ametC9xeuIet3rMudEnVvcJCtt8PN+l19eGDPfA65RFB/D1gNUX/aIrkN4d3fLxzjLfR/fx1Alc14FHteK57hy8h0WEvRQJssaIKFcjRhZraLW8O9DiRtELx+xXJcolNDOu85gEHXV9xwR9lkvMgGAwcd9xLaH5tYeQoaCmxUZ4qZlvWKyw1IyC+6p+0BXIrRabbsOP20zmJKVY0rJxFTMnkeA1jo4c6z2upoThonwoEwwwDcVXRIAN1ElmhvgKrY8vp4FH7eVkdh9Cn0JXpW9OnRcNujrqRF2+RDhDNxuLupEYYCzcje5S3c07dsg2rnaQ37iGxFxQJ3YUvJWg50ddMbGuQU0syXq9BjN1sZnLJeQDLByh2T9X8WgB8HmU0DFxV0RNvJpKylyYNxUtLUQj6iH3czwYp31sH6Nj9PWhV1TQpYeme4eNLNEvOH9JQyfwnsIhwf5bbky010sNOmg7JaNS6aeSWnV3XBJ42F1UqclYlVCYgqExreai6J9wExSVFb23nrAQHcRMX8DOYMQH+Jl1K5gSOek3sVdZVPAq7/8EYSIJJdwoGGD50p/+1gyNoTuIgP/DhHHuzn99DD1ylBxmeZhU0Ef7ygfjU5H7IWJEsswssp+mS9VQ0Vh57+bY1I6TW4Iai/vjMx171J7p5IxX2SpZVh+W620Uwsf1tv12cf722u6j8Pp5szm/Ok8eNxsQP+++VfvdE4DNZm0Em/NLM79Z5j+a37QHnJ7vtp/1tLU/uS6Ko8m/tfttu3HT68Om2H1ti/Zlt/9TT/yyhv+/wvQX8+va6tPb9umw3m3PnttP7/4Lls6edtvXw+q1PVy852laryDcV+To9ROs5uLX5fb56l+DKyMtl9Wl8wXlhnM97aNV8J/rL49Pv12c5+eXV4PLszVYeNw+tav26+Pmr7ettv1eG//fWQse6M/dfv0Z7OigXfz6DQy+tK+vj5/bq2/V4/7x5Wpj/lye7dvD2377jnbCc0Db/9FEa+gfGFmW6fXf9cuqX620SysY+UVP+XLYr0Dblf4LMTi0P/QRRn7QRv+mjx/g33X7++ani8P+rb38ib134NPj5hVIoEl/t6xHH/XYw261Ptqrk8Ho9kdmx61hHMMFw7//A92UHZA=\".unpack('m0')[0]))");
		pRGSSEval("module Graphics;def self.focus;@focus;end;def self.focus=(b);@focus=b;end;end");
		pRGSSEval("Graphics.focus=true");

        // 4、主逻辑
        pRGSSGameMain(hWnd, szScripts, (pRgssad ? (char**)pRgssad : &pRgssad));
 
__exit:
 
        if (hRgssCore)
        {
                FreeLibrary(hRgssCore);
                hRgssCore = NULL;
        }
 
        if (hWnd)
        {
                DestroyWindow(hWnd);
                hWnd = NULL;
        }
 
        UnregisterClassA(pWndClassName, hInstance);
 
        return 0;
}

LRESULT CALLBACK BindWndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{

	switch(msg)
    {
	case WM_INITMENUPOPUP:
		if (lParam | 0xFFFF0000 != 0)
			SendMessage(hWnd, WM_CANCELMODE, 0, 0);
			return 0;
	case WM_SYSKEYDOWN:
		switch(wParam)
		{
		case VK_MENU:
			//keybd_event(0xA4, 0, 0, 0);
			//keybd_event(0xA4, 0, 2, 0);
			return 0;
		}
	case WM_SYSKEYUP:
		switch(wParam)
		{
		case VK_MENU:
			//keybd_event(0xA4, 0, 0, 0);
			//keybd_event(0xA4, 0, 2, 0);
			return 0;
		}
    case WM_SETFOCUS:
		ShowCursor(0);
        pRGSSEval("Graphics.focus=true");
        return 0;
    case WM_KILLFOCUS:
		ShowCursor(1);
        pRGSSEval("Graphics.focus=false");
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return(DefWindowProc(hWnd, msg, wParam, lParam));
}
