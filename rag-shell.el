;;; rag-shell.el --- Retrieval Augumented Generation app for emacs -*- lexical-binding: t -*-

;; Copyright (C) 2024 tomoyukim <tomoyukim@outlook.com>
;; Author: Tomoyuki Murakami
 ;; URL: https://github.com/tomoyukim/rag-shell.el
;; Created: 2024
;; Package-Version: 20240505.1
;; Version: 0.2
;; Homepage: https://github.com/tomoyukim/rag-shell
;; Keywords: org-mode rag chatgpt llm
;; Package-Requires: ((epc))

;; This file is NOT part of GNU Emacs.

;; This program is free software; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation; either version 3, or (at your option)
;; any later version.
;;
;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.
;;
;; You should have received a copy of the GNU General Public License
;; along with GNU Emacs; see the file COPYING.  If not, write to the
;; Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
;; Boston, MA 02110-1301, USA.

;;; Commentary:

;; This package experimentally provides a standard RAG application.


;;; Code:
(require 'epc)

(defcustom rag-shell-chat-model-temperature 0.7
  "Temperature for LLM model used in chat"
  :type 'float
  :group 'rag-shell)

(defcustom rag-shell-openai-key nil
  "OpenAI key as a string or a function that loads and returns it."
  :type '(choice (function :tag "Function")
                 (string :tag "String"))
  :group 'rag-shell)

(defcustom rag-shell-embedding-model '("huggingface" . "intfloat/multilingual-e5-large")
  ""
  :type '((cons
           (string :tag "String")
           (string :tag "String")))
  :group 'rag-shell)

(defcustom rag-shell-chat-model '("openai" . "gpt-4-0125-preview")
  ""
  :type '((cons
           (string :tag "String")
           (string :tag "String")))
  :group 'rag-shell)

 (defcustom rag-shell-rag-source-list '()
  ""
  :type '((list
           (cons
            (string :tag "String")
            (string :tag "String"))))
  :group 'rag-shell)


(defvar rag-shell-server nil)
(defvar rag-shell-current-topic nil)

(defun -rag-shell-openai-key ()
  "Get the OpenAI API key."
  (cond ((stringp rag-shell-openai-key)
         rag-shell-openai-key)
        ((functionp rag-shell-openai-key)
         (condition-case _err
             (funcall rag-shell-openai-key)
           (error
            "KEY-NOT-FOUND")))
        (t
         nil)))

(defun -rag-shell-change-topic ()
  ""
  (let* ((src-names (mapcar #'car rag-shell-rag-source-list))
         (selected-src-name (completing-read "Select source: " src-names))
         (selected-src (assoc selected-src-name rag-shell-rag-source-list))
         (src-name (car selected-src))
         (src (cdr selected-src)))
    (setq rag-shell-current-topic src-name)
    (epc:call-deferred rag-shell-server 'setup_rag_chain `(,src-name ,src))))

;;;###autoload
  (defun rag-shell-start ()
    ""
    (interactive)
    (setq rag-shell-server (epc:start-epc "python" '("rag-shell-server.py")))
    ;; TODO: error handling
    (let ((key (-rag-shell-openai-key))
          (emb-model-type (car rag-shell-embedding-model))
          (emb-model (cdr rag-shell-embedding-model))
          (model-type (car rag-shell-chat-model))
          (model (cdr rag-shell-chat-model))
          (temperature rag-shell-chat-model-temperature))
      (deferred:$
       (epc:call-deferred rag-shell-server 'setup_embedding_model `(,emb-model-type ,emb-model))
       (deferred:nextc it
                       (lambda ()
                         (epc:call-deferred rag-shell-server 'setup_chat_model `(,model-type ,model ,temperature ,key))))
       (deferred:nextc it
                       (lambda ()
                         (-rag-shell-change-topic)
                         ))
       (deferred:nextc it
                       (lambda ()
                         (message (format "rag-shell-server is ready.(%s)" rag-shell-current-topic))))))
    )


;;;###autoload
(defun rag-shell-stop ()
  ""
  (interactive)
  (epc:stop-epc rag-shell-server)
  (setq rag-shell-server nil))

;;;###autoload
(defun rag-shell-change-topic ()
  ""
  (interactive)
  (deferred:$
   (-rag-shell-change-topic)
   (deferred:nextc it
                   (lambda ()
                     (message (format "rag-shell topic is changed to %s" rag-shell-current-topic))))))

;;;###autoload
(defun rag-shell-ask (query)
  ""
  (interactive "sQuestion: ")
  (if rag-shell-server
      (deferred:$
       (epc:call-deferred rag-shell-server 'chat `(,query))
       (deferred:nextc it
                       (lambda (res)
                         (let ((buffer (get-buffer-create "*rag-shell-answer*")))
                           (with-current-buffer buffer
                             (erase-buffer)
                             (insert (concat "* " query "\n"))
                             (insert res)
                             (insert (format "Answered at: %s\n\n" (current-time-string)))
                             (org-mode)
                             (view-mode +1)
                             (setq view-exit-action 'kill-buffer))
                           (display-buffer buffer)))))
    (message "rag-shell-server is not started")))


(provide 'rag-shell)

;;; rag-shell.el ends here
