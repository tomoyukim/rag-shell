# rag-shell
This package is experimental project to integrate standard RAG application for Emacs powered by LangChainin. This is in very early stage.

## Config
```elisp
(use-package rag-shell
  :config
  (setq epc:debug-out t)
  :custom
  (rag-shell-rag-source-list '(("elisp_ja_29.3" . "https://ayatakesi.github.io/lispref/29.3/elisp-ja.html")))
  (rag-shell-openai-key (lambda ()
                          (auth-source-pass-get 'secret "openai-key"))))

```