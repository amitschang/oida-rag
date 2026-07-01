//! Interactive REPL and one-shot entry points for the CLI.

use anyhow::Context;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

use super::agent::Agent;

/// Run a single query and print the answer.
pub async fn run_once(agent: &Agent, query: String) -> anyhow::Result<()> {
    let mut history = agent.new_history();
    let answer = agent.ask(&mut history, query).await?;
    println!("\n{answer}");
    Ok(())
}

/// Run the interactive read-eval-print loop. `label` brands the banner and
/// prompt (e.g. `OIDA` → `OIDA assistant…` / `oida> `).
pub async fn run_repl(agent: &Agent, label: &str) -> anyhow::Result<()> {
    let mut editor = DefaultEditor::new().context("initializing line editor")?;
    let mut history = agent.new_history();

    println!("{label} assistant. Ask about documents and their connections.");
    println!("Commands: /reset to clear context, /exit or Ctrl-D to quit.\n");

    let prompt = format!("{}> ", label.to_lowercase());
    loop {
        match editor.readline(&prompt) {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }
                let _ = editor.add_history_entry(input);

                match input {
                    "/exit" | "/quit" => break,
                    "/reset" => {
                        history = agent.new_history();
                        println!("(context cleared)\n");
                        continue;
                    }
                    _ => {}
                }

                match agent.ask(&mut history, input.to_string()).await {
                    Ok(answer) => println!("\n{answer}\n"),
                    Err(e) => eprintln!("error: {e:#}\n"),
                }
            }
            Err(ReadlineError::Interrupted) => continue,
            Err(ReadlineError::Eof) => break,
            Err(e) => return Err(e).context("reading input"),
        }
    }

    println!("bye");
    Ok(())
}

/// Print a short banner describing the connected tools.
pub fn print_tools(tools: &[rmcp::model::Tool], label: &str) {
    eprintln!("Connected to {label} MCP server. Tools available:");
    for t in tools {
        let desc = t
            .description
            .as_ref()
            .map(|d| d.split('.').next().unwrap_or(d).to_string())
            .unwrap_or_default();
        eprintln!("  - {}: {desc}", t.name);
    }
    eprintln!();
}
