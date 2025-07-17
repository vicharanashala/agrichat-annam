
If you encounter issues while setting up or running AgriChat-Annam, refer to the table below for common problems and their solutions. These tips should help you resolve most setup and runtime errors quickly.

| Issue                   | Solution                                                                 |
|-------------------------|--------------------------------------------------------------------------|
| Ollama not running      | Run `ollama serve` in a separate terminal window.                        |
| Model not found         | Verify with `ollama list` and pull required models using `ollama pull`.  |
| ChromaDB path errors    | Use absolute paths for Windows systems (e.g., `C:\\path\\to\\chroma_db`).|
| Low memory errors       | Use smaller models like `gemma:1b` or reduce the ChromaDB size.          |

> **Tip:**  
> If you see errors not listed here, check your terminal output for detailed messages.  
> - For Ollama issues, check the logs in `~/.ollama/logs/server.log` (Mac/Linux) or `%LOCALAPPDATA%\Ollama\server.log` (Windows).
> - For persistent problems, restarting your machine or reinstalling dependencies can often help.

If you need more help, consult the official documentation for [Ollama](https://github.com/ollama/ollama/blob/main/docs/troubleshooting.md) and [ChromaDB](https://cookbook.chromadb.dev/faq/).
