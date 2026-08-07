from jarvis_platform.db.models.voice_foundation_model import (
    UserIdentityModel,
    VoiceConsentModel,
    VoicePreferenceModel,
    VoiceSessionModel,
    VoiceProfileModel,
    SpeakerEventModel,
)


def test_voice_foundation_models_store_metadata_not_biometrics() -> None:
    assert UserIdentityModel.__tablename__ == "users"
    assert VoiceSessionModel.__tablename__ == "voice_sessions"
    assert VoicePreferenceModel.__tablename__ == "voice_preferences"
    assert VoiceConsentModel.__tablename__ == "voice_consents"
    assert VoiceProfileModel.__tablename__ == "voice_profiles"
    assert SpeakerEventModel.__tablename__ == "speaker_events"
    assert UserIdentityModel.__table__.schema == "identity"
    assert VoiceSessionModel.__table__.schema == "voice"
    assert VoicePreferenceModel.__table__.schema == "voice"
    assert VoiceConsentModel.__table__.schema == "voice"
    assert VoiceProfileModel.__table__.schema == "voice"
    assert SpeakerEventModel.__table__.schema == "voice"

    columns = {
        column.name
        for model in (
            UserIdentityModel,
            VoiceSessionModel,
            VoicePreferenceModel,
            VoiceConsentModel,
            VoiceProfileModel,
            SpeakerEventModel,
        )
        for column in model.__table__.columns
    }
    assert "audio" not in columns
    assert "embedding" not in columns
    assert "voice_profile" not in columns
