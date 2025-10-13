import type { ChangeEvent, FormEvent } from "react";
import type { DatabaseToggleConfig } from "../types/chat";

interface ComposerProps {
  question: string;
  onQuestionChange: (value: string) => void;
  onSubmit: () => void;
  disabled?: boolean;
  stateValue: string;
  onStateChange: (value: string) => void;
  language: string;
  onLanguageChange: (value: string) => void;
  toggles: DatabaseToggleConfig;
  onToggleChange: (partial: Partial<DatabaseToggleConfig>) => void;
}

export function Composer({
  question,
  onQuestionChange,
  onSubmit,
  disabled,
  stateValue,
  onStateChange,
  language,
  onLanguageChange,
  toggles,
  onToggleChange,
}: ComposerProps) {
  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    onSubmit();
  };

  const handleCheckbox = (event: ChangeEvent<HTMLInputElement>) => {
    const { name, checked } = event.target;
    onToggleChange({ [name]: checked } as Partial<DatabaseToggleConfig>);
  };

  return (
    <form className="composer" onSubmit={handleSubmit}>
      <div className="composer__inputs">
        <textarea
          value={question}
          onChange={(event) => onQuestionChange(event.target.value)}
          placeholder="Ask about crops, soil, pests, fertilizers..."
          rows={2}
          disabled={disabled}
          required
        />
        <div className="composer__controls">
          <div className="composer__field">
            <label htmlFor="state">State</label>
            <input
              id="state"
              type="text"
              value={stateValue}
              onChange={(event) => onStateChange(event.target.value)}
              placeholder="e.g. Haryana"
              disabled={disabled}
            />
          </div>
          <div className="composer__field">
            <label htmlFor="language">Language</label>
            <input
              id="language"
              type="text"
              value={language}
              onChange={(event) => onLanguageChange(event.target.value)}
              placeholder="English"
              disabled={disabled}
            />
          </div>
        </div>
      </div>
      <div className="composer__toggles">
        <fieldset>
          <legend>Knowledge Sources</legend>
          <label>
            <input
              type="checkbox"
              name="golden_enabled"
              checked={toggles.golden_enabled}
              onChange={handleCheckbox}
              disabled={disabled}
            />
            Golden Database
          </label>
          <label>
            <input
              type="checkbox"
              name="pops_enabled"
              checked={toggles.pops_enabled}
              onChange={handleCheckbox}
              disabled={disabled}
            />
            PoPs Database
          </label>
          <label>
            <input
              type="checkbox"
              name="llm_enabled"
              checked={toggles.llm_enabled}
              onChange={handleCheckbox}
              disabled={disabled}
            />
            LLM Fallback
          </label>
        </fieldset>
        <fieldset>
          <legend>Display</legend>
          <label>
            <input
              type="checkbox"
              name="show_database_path"
              checked={toggles.show_database_path}
              onChange={handleCheckbox}
              disabled={disabled}
            />
            Show database path
          </label>
          <label>
            <input
              type="checkbox"
              name="show_confidence_scores"
              checked={toggles.show_confidence_scores}
              onChange={handleCheckbox}
              disabled={disabled}
            />
            Show confidence scores
          </label>
          <label>
            <input
              type="checkbox"
              name="enable_adaptive_thresholds"
              checked={toggles.enable_adaptive_thresholds}
              onChange={handleCheckbox}
              disabled={disabled}
            />
            Adaptive thresholds
          </label>
          <label>
            <input
              type="checkbox"
              name="strict_validation"
              checked={toggles.strict_validation}
              onChange={handleCheckbox}
              disabled={disabled}
            />
            Strict validation
          </label>
        </fieldset>
        <button type="submit" className="primary" disabled={disabled || !question.trim()}>
          Send
        </button>
      </div>
    </form>
  );
}
